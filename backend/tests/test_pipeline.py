import unittest
from unittest.mock import Mock, patch

from backend.app.guardrails import verify_answer_grounding
from backend.app import vector_store
from backend.app.conversation_memory import (
    clear_history,
    get_history,
    save_message
)
from backend.app.rag import generate_answer, rewrite_followup_question


class GroundingVerificationTests(unittest.TestCase):
    @patch("backend.app.guardrails.ollama.chat")
    def test_accepts_supported_verdict_with_common_formatting(self, chat):
        chat.return_value = {
            "message": {"content": "Verdict: SUPPORTED. The facts match."}
        }

        self.assertTrue(
            verify_answer_grounding(
                "The policy begins in June.",
                "The policy starts in June."
            )
        )

    @patch("backend.app.guardrails.ollama.chat")
    def test_accepts_verdict_at_end_of_explanation(self, chat):
        chat.return_value = {
            "message": {
                "content": (
                    "The answer is SUPPORTED. It matches the evidence."
                )
            }
        }

        self.assertTrue(
            verify_answer_grounding(
                "The policy begins in June.",
                "The policy starts in June."
            )
        )

    @patch("backend.app.guardrails.ollama.chat")
    def test_rejects_unsupported_or_ambiguous_verdicts(self, chat):
        for verdict in (
            "RETURN UNSUPPORTED. The month is not in the evidence.",
            "I think this is supported",
            '{"verdict":"UNCERTAIN"}',
            "Verdict: SUPPORTED, but the context is insufficient."
        ):
            with self.subTest(verdict=verdict):
                chat.return_value = {
                    "message": {"content": verdict}
                }

                self.assertFalse(
                    verify_answer_grounding(
                        "The policy begins in July.",
                        "The policy starts in June."
                    )
                )

    @patch("backend.app.guardrails.ollama.chat")
    def test_requests_deterministic_structured_verdict(self, chat):
        chat.return_value = {
            "message": {"content": '{"verdict":"SUPPORTED"}'}
        }

        self.assertTrue(
            verify_answer_grounding(
                "The policy begins in June.",
                "The policy starts in June."
            )
        )
        self.assertEqual(chat.call_args.kwargs["format"], "json")
        self.assertEqual(
            chat.call_args.kwargs["options"],
            {"temperature": 0, "seed": 0, "num_predict": 32}
        )

    @patch("backend.app.guardrails.ollama.chat")
    def test_accepts_a_direct_evidence_phrase_without_llm_verification(self, chat):
        context = (
            "The Acme sample record uses unique identifier "
            "Marigold 418 and category saffron."
        )
        answer = (
            "The unique identifier written in this PDF "
            "is Marigold 418."
        )

        self.assertTrue(verify_answer_grounding(answer, context))
        chat.assert_not_called()

    @patch("backend.app.guardrails.ollama.chat")
    def test_verifier_receives_question_and_ignores_omitted_facts(self, chat):
        chat.return_value = {
            "message": {
                "content": (
                    "Based on the comparison, I return SUPPORTED."
                )
            }
        }
        question = "What category does the Acme sample record have?"

        self.assertTrue(
            verify_answer_grounding(
                "The category of the Acme sample record is saffron.",
                (
                    "The Acme sample record uses unique identifier "
                    "Marigold 418 and category saffron."
                ),
                question=question
            )
        )
        prompt = chat.call_args.kwargs["messages"][0]["content"]
        self.assertIn(question, prompt)
        self.assertIn("Do not require", prompt)

    @patch("backend.app.guardrails.ollama.chat")
    def test_accepts_extractive_summary_bullets(self, chat):
        chat.return_value = {
            "message": {"content": '{"verdict":"SUPPORTED"}'}
        }
        context = (
            "The Acme sample record uses unique identifier "
            "Marigold 418 and category saffron."
        )
        answer = (
            "Here are the key facts extracted from the document text:\n"
            "- The Acme sample record uses unique identifier Marigold 418.\n"
            "- The Acme sample record uses category saffron."
        )

        self.assertTrue(
            verify_answer_grounding(
                answer,
                context,
                question="List the key facts in the document."
            )
        )
        chat.assert_called_once()

    @patch("backend.app.guardrails.ollama.chat")
    def test_does_not_accept_a_supported_value_with_the_wrong_label(self, chat):
        chat.return_value = {
            "message": {"content": "UNSUPPORTED"}
        }

        self.assertFalse(
            verify_answer_grounding(
                "The category is Marigold 418.",
                (
                    "The Acme sample record uses unique identifier "
                    "Marigold 418 and category saffron."
                ),
                question="What is the category?"
            )
        )

    @patch("backend.app.guardrails.ollama.chat")
    def test_does_not_combine_evidence_from_different_records(self, chat):
        chat.return_value = {
            "message": {"content": '{"verdict":"UNSUPPORTED"}'}
        }

        self.assertFalse(
            verify_answer_grounding(
                "Alice's title is director.",
                "Alice's title is manager. Bob's title is director."
            )
        )
        chat.assert_called_once()


class ChromaRetrievalTests(unittest.TestCase):
    @patch("backend.app.vector_store.generate_query_embedding")
    def test_semantic_search_does_not_read_the_whole_collection(self, embed):
        embed.return_value = [0.25, 0.75]
        collection = Mock()
        collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Relevant text"]],
            "metadatas": [[{"document_id": 7, "chunk_index": 0}]]
        }

        with patch.object(vector_store, "collection", collection):
            results = vector_store.semantic_search(
                "relevant",
                top_k=10,
                document_id=7
            )

        collection.query.assert_called_once_with(
            query_embeddings=[[0.25, 0.75]],
            n_results=10,
            include=["documents", "metadatas"],
            where={"document_id": 7}
        )
        collection.get.assert_not_called()
        self.assertEqual(results[0]["id"], "chunk-1")

    @patch("backend.app.vector_store.generate_embeddings")
    def test_chunk_storage_uses_bounded_upsert_batches(self, embed):
        embed.side_effect = lambda texts: [
            [float(index)] for index, _ in enumerate(texts)
        ]
        collection = Mock()
        chunks = [f"chunk {index}" for index in range(501)]

        with patch.object(vector_store, "collection", collection):
            stored = vector_store.store_chunks(chunks, 9)

        self.assertEqual(stored, 501)
        self.assertEqual(collection.upsert.call_count, 2)
        first_batch = collection.upsert.call_args_list[0].kwargs
        second_batch = collection.upsert.call_args_list[1].kwargs
        self.assertEqual(len(first_batch["ids"]), 500)
        self.assertEqual(second_batch["ids"], ["doc_9_chunk_500"])
        self.assertEqual(
            second_batch["metadatas"],
            [{"document_id": 9, "chunk_index": 500}]
        )


class FollowUpMemoryTests(unittest.TestCase):
    @patch(
        "backend.app.rag.generate_with_ollama",
        return_value="What category does the Acme sample record have?"
    )
    def test_follow_up_rewrite_receives_prior_document_evidence(self, model):
        session_id = "follow-up-evidence-test"
        document_id = 982341
        evidence = (
            "The Acme sample record uses unique identifier "
            "Marigold 418 and category saffron."
        )

        try:
            save_message(
                session_id,
                "What is the identifier?",
                "Marigold 418",
                document_id,
                evidence=evidence
            )
            history = get_history(session_id, document_id)
            rewritten = rewrite_followup_question(
                "What category does it have?",
                history
            )

            self.assertIn("Acme sample record", rewritten)
            prompt = model.call_args.args[0]
            self.assertIn(evidence, prompt)
        finally:
            clear_history(session_id, document_id)

    @patch("backend.app.rag.save_message")
    @patch(
        "backend.app.rag.verify_answer_grounding",
        return_value=True
    )
    @patch(
        "backend.app.rag.generate_with_ollama",
        return_value="The sample record's category is saffron."
    )
    @patch(
        "backend.app.rag.get_neighboring_chunks",
        side_effect=lambda results, window: results
    )
    @patch(
        "backend.app.rag.rerank_results",
        side_effect=lambda question, results, top_k: results
    )
    @patch(
        "backend.app.rag.hybrid_search",
        return_value=[{
            "id": "chunk-1",
            "text": (
                "The Acme sample record uses unique identifier "
                "Marigold 418 and category saffron."
            ),
            "metadata": {"document_id": 982341, "chunk_index": 0}
        }]
    )
    @patch(
        "backend.app.rag.rewrite_followup_question",
        return_value="What category does the Acme sample record have?"
    )
    @patch("backend.app.rag.get_history", return_value=[{"role": "user"}])
    def test_follow_up_verifies_answer_against_retrieved_context(
        self,
        history,
        rewrite,
        search,
        rerank,
        neighbors,
        generate,
        verify,
        save
    ):
        retrieved_context = (
            "The Acme sample record uses unique identifier "
            "Marigold 418 and category saffron."
        )

        response = generate_answer(
            "What category does it have?",
            "follow-up-rag-test",
            document_id=982341
        )

        verify.assert_called_once_with(
            "The sample record's category is saffron.",
            retrieved_context,
            question="What category does the Acme sample record have?"
        )
        self.assertEqual(response["answer"], "The sample record's category is saffron.")


if __name__ == "__main__":
    unittest.main()
