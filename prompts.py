from langchain.prompts import PromptTemplate


reducer_templates = {

    "key_terms_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for extracting:
- keywords
- topics
- themes
- key concepts
- definitions

Instructions:
- Remove examples, anecdotes, references, tables, and citations.
- Remove formatting, headers, footers, page numbers, and boilerplate.
- Keep only sentences that define, explain, or introduce important technical terms and concepts.
- Preserve technical vocabulary exactly as written.
- Do not summarize; only filter and condense.
- Maintain original meaning and terminology.

Output:
Return the reduced technical text only (no commentary, no JSON).

TEXT:
{text}
"""
    ),


    "summary_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for generating:
- the main topic
- a core explanation using the 80/20 rule

Instructions:
- Remove minor details, citations, tables, and side discussions.
- Keep only sentences that explain what the topic is and how it works.
- Preserve causal relationships and core arguments.
- Ignore formatting noise such as headers, footers, and page numbers.
- Preserve important technical vocabulary exactly as written.
- Do not add new information or interpretations.

Output:
Return a concise reduced version of the text containing only the essential explanatory content.

TEXT:
{text}
"""
    ),


    "conclusion_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for extracting:
- TLDR
- key takeaways
- conclusion

Instructions:
- Remove background theory, methodology details, and examples.
- Keep only sentences that express outcomes, implications, and high-level insights.
- Retain statements about importance, impact, or lessons learned.
- Ignore formatting noise such as headers, footers, and page numbers.
- Preserve technical vocabulary exactly as written.
- Do not summarize; only filter to conclusion-relevant content.

Output:
Return the reduced text focused only on takeaways and conclusions.

TEXT:
{text}
"""
    ),


    "academic_reducer": PromptTemplate(
        input_variables=["text"],
        template="""
You are a text reduction assistant.

Your task is to compress the following document while preserving only information useful for extracting:
- bibliographic metadata
- research question or objective
- methodology
- key findings or results
- limitations or assumptions

Instructions:
- Remove discussion, background theory, examples, and unrelated narrative.
- Keep only sentences that describe:
  - what problem is studied
  - how the study was conducted
  - what was discovered
  - weaknesses or constraints
- Preserve technical vocabulary exactly as written.
- Ignore formatting noise such as headers, footers, page numbers, and references list formatting.
- Do not interpret or summarize beyond filtering.

Output:
Return the reduced research-focused text only (no JSON, no commentary).

TEXT:
{text}
"""
    )
}

student_templates = {
    "key_terms": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).

        Identify and extract:

        Keywords – important non-trivial words and phrases.

        Topics – major subject areas discussed.

        Themes – higher-level conceptual ideas that unify the document.
        
        Key Concepts - important concepts that are essential to know 
        
        For each key term, provide a concise definition (1-2 sentences) that explains its meaning in the context of the research.


        Keep all extracted items concise and meaningful.


Format the output as valid JSON:

{{
    "keywords": [{{"term":"...","definition":"..."}}],
    "primary_topics": [{{"topic":"...","explanation":"..."}}],
    "concepts": [{{"concepts":"...","explanation":"..."}}],
    "themes": [{{"theme":"...","explanation":"..."}}]
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.

Text:
{text}
"""
    ),
    "summary": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).
        
        Identify the central topic and purpose of the document.
        2. Extract only the most important ideas that account for the majority of understanding (Pareto principle).
        3. Present the material as if introducing it to a student encountering this topic for the first time.
        Identify and extract:

        Introduction- purpose, motivation, and overall aim(4 sentences)
        Main topic - What the topic is about
        Core Explanation – 4-5 sentences explaining the topic using the 80/20 rule
        

        Keep all extracted items concise and meaningful.

Format the output as valid JSON:

{{
    "introduction":"...",
    "main topic": "...",
    "explanation": "...",
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.

Text:
{text}
"""
    ),
    "conclusion": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Given the following section, 
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).

        Identify and extract:

        TLDR – important non-trivial words and phrases.

        Key takeaways – major subject areas discussed.

        Conclusion – higher-level conceptual ideas that unify the document.
        

        Keep all extracted items concise and meaningful.


Format the output as valid JSON:

{{
    "key_takeaways": [{{"takeaway":"...","explanation":"..."}}],
    "tldr": {{"..."}},
    "conclusion":{{ "..."}},
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.

Text:
{text}
"""
    ),
    "academic": PromptTemplate(
        input_variables=['text'],
        template="""
        You are an expert NLP and document analysis assistant.
        Your task is to analyze the provided Markdown or PDF text and extract structured semantic information.

        Instructions:

        Parse the input document content accurately (ignore formatting noise such as headers, page numbers, or footers).

        Identify and extract:

        Bibliographic metadata: (title, authors, year, source, URL)1. 
        
        In-text citations or reference markers (e.g., [1], (Smith, 2022), etc.)

        Research question or objective - problem being investigated and overall hypothesis

        Methodology used - how they arrived at their results

        Key findings/results - the main results that were discovered

        Limitations or assumptions - what might be wrong or incomplete

        Keep all extracted items concise and meaningful.


Format the output as valid JSON:

{{
    "metadata": {
    "title": "...",
    "authors": [...],
    "year": "...",
    "source": "...",
    "url": "...",
    "citations":[{{"citation": "..."}}]
    },
  "research_question": "...",
  "methodology": "...",
  "key_findings": [{{"finding":"...","description":"..."}}],
  "limitations": [{{"limitation":"...","description":"..."}}],
}}

Constraints:

Preserve important technical vocabulary exactly as written.
Only include terms that are specific, non-generic, and relevant to the technical context.


Text:
{text}
"""
    )
    
}
