# ------------------ RAG Query & Execution (REPLACE your existing query block) ------------------
query = st.text_input("Ask a question:")

if query:
    if not os.path.exists(file_path):
        st.error("Vector store not found. Please process data first.")
        st.stop()

    # Load vectorstore
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    # Make a retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # Step 1: retrieve documents (wrap to catch embed errors)
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        # This is likely coming from the embedding library (sentence-transformers / torch)
        st.error("Failed while computing embeddings for the query. See diagnostics below.")
        st.write("Error type:", type(e).__name__)
        st.write(str(e))
        st.info(
            "Common causes: incompatible/pinned versions of sentence-transformers, transformers, or torch. "
            "See the suggested requirements.txt in the app root."
        )
        st.stop()

    if not docs:
        st.warning("No relevant documents found. Try processing more sources or check your inputs.")
    else:
        # Build a context string (limit to top 5 docs and first 1000 chars each to avoid huge prompts)
        top_docs = docs[:5]
        context_parts = []
        for i, d in enumerate(top_docs, start=1):
            text_snippet = (d.page_content[:1000] + "...") if len(d.page_content) > 1000 else d.page_content
            # include some metadata if present
            meta = d.metadata if getattr(d, "metadata", None) else {}
            source = meta.get("source") or meta.get("url") or meta.get("file_path") or f"doc_{i}"
            context_parts.append(f"Source: {source}\n\n{text_snippet}")

        context = "\n\n---\n\n".join(context_parts)

        # Prompt template: use your template but ensure placeholders match
        TEMPLATE = """
You are an expert research assistant. Use ONLY the provided context to answer concisely.
If the answer is not present in the context, say "I don't know."

Context:
{context}

Answer:
"""
        prompt_text = TEMPLATE.format(question=query, context=context)

        # Call the LLM (try .invoke first, fallback to .generate)
        try:
            # Many lightweight wrappers accept a simple string
            response = llm.invoke(prompt_text)
            # If llm.invoke returns a dict/object, try to extract text
            if isinstance(response, dict):
                out = response.get("text") or response.get("answer") or str(response)
            else:
                out = str(response)
        except Exception as e_invoke:
            # fallback to .generate (common LangChain pattern)
            try:
                gen = llm.generate([prompt_text])
                # extract text from generation object (many implementations store generations[0][0].text)
                out = None
                if hasattr(gen, "generations"):
                    g0 = gen.generations[0][0]
                    out = getattr(g0, "text", str(g0))
                else:
                    out = str(gen)
            except Exception as e_gen:
                st.error("LLM call failed. See error details below.")
                st.write("invoke error:", type(e_invoke).__name__, str(e_invoke))
                st.write("generate error:", type(e_gen).__name__, str(e_gen))
                st.stop()

        st.header("Answer")
        st.write(out)

        st.subheader("Context / Source snippets used")
        for i, d in enumerate(top_docs, start=1):
            meta = d.metadata if getattr(d, "metadata", None) else {}
            source = meta.get("source") or meta.get("url") or meta.get("file_path") or f"doc_{i}"
            st.write(f"**Source {i}:** {source}")
            st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))
