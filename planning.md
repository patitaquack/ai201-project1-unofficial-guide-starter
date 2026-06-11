# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

"How to get help for your mental health at Utep?"

When a student is experiencing mental health issues such as anxiety, depression, or feeling overwhelmed, it can be hard to know where to find help and get it in a timely manner. There are many resources available for students at UTEP. 
---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | url    |Virtual Care | https://timelycare.com/utep/|
| 2 | url|Counseling Services|https://www.utep.edu/student-affairs/counsel/resources/services-students.html |
| 3 |url | student support|https://www.utep.edu/student-affairs/miner-support/ |
| 4 |forum threads |inquiry|https://www.reddit.com/r/UTEP/comments/t9la07/how_does_mental_health_services_work/ |
| 5 | pdf | referral list |https://www.utep.edu/student-affairs/counsel/_files/docs/community-referral-book-2025-20261.pdf |
| 6 |pdf | depression help|https://www.utep.edu/student-affairs/counsel/_files/docs/depression.pdf |
| 7 | pdf| test anxiety|https://www.utep.edu/student-affairs/counsel/_files/docs/test-anxiety.pdf|
| 8 |pdf | substance abuse help|https://www.utep.edu/student-affairs/counsel/_files/docs/problem-with-alcohol-or-drugs.pdf |
| 9 | pdf| stress management help|https://www.utep.edu/student-affairs/counsel/_files/docs/stress-management.pdf |
| 10 |url | online counsel resources | https://www.utep.edu/student-affairs/counsel/resources/on-line-resources.html|

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** ~ 200 tokens ( about 800 chars). One entry per provider in the referral list, one comment per Reddit thread. 

**Overlap:** ~50 tokens  for the prose handouts ( small articles or specific sections)
and service web pages. Overlap = 0 for the referral list. Online-Resources
link list, where each entry is self-contained.

**Reasoning:** My sources are short. Structured "hand-outs" or small articles for depression, anxiety, and stress. Including a provider directory, and a forum thread. Semantic boundaries are important.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** sentence-transformers (all-MiniLM-L6-v2)

**Top-k:** 3. My corpus is small, about 10 curated sources. 
Each query is a student trying to find the single most relevant resource, so returning
the 3 closest chunks keeps results focused and easily fits in the LLM prompt.


**Production tradeoff reflection:**
Because Utep is a Univeristy in a border city, I would definitely make this bilingual. Upgrading to multilingual model would be the first thing I would consider. I would also like to include more sources like organizations that focus on mental health or overall health. Also adding more recources to self help. I would consuder  moving to a premium API
model if evaluation showed accuracy still falling short.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 |I keep getting really anxious and forgetting everything I have studied during exams — does UTEP have resources for students like me?  | Here is the information on counseling services at Utep, and also advice on how to manage test anxiety from home. |
| 2 |Can I get free counseling at Utep? | Utep offers free counseling for students and staff, there is also timely care offered which you can access from home. |
| 3 | I have been feeling depressed, and I would like to know if Utep offers anything to help me. |Utep offers counceling services, as well as certain events and practices that can help with depression. |
| 4 |I feel overwhelmed, and have so much stress, what services does Utep offer that may help with stress? | Utep offers in person counseling, at home virtual services, meditation events at Utep.  |
| 5 | If I don't want to see a counselour in person, how can I get help for my mental health as a Utep student? |As a student you can access timely care which is a virtual program where you can speak to a certified therapist from the comfort of your home. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Noisy results, specially with PDF's, If it is not extracted right it could output mixed information, unrelated.

2. Reddit thread, Because it is coming from other former or actual students, it may offer false information, it could also give half responses if chunking is not the right size.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

![Pipeline diagram](images/Mermaid.png)


---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

i will be using Claude, to help me, I will share my chunking plan and my pipeline. I will ask for suggestions on how accurate output will be with my chunk size and if my plan will result in succesful responses. 
I expect Claude to correct my sizes to better chunk pieces. 
I will verify by making sure that chunk does not exceed 256 limit. And that information like the referral lists are outputted corrrectly. 


**Milestone 3 — Ingestion and chunking:**

I'll give Claude my Documents section and my Chunking plan (~200-token chunks and  overlap rules). 
 I expect for it to  suggest corrections to my chunk size if needed. I'll verify
by checking that no chunk exceeds the model's 256-token limit, that the referral-list entries come out whole (each provider's name kept with their contact info), and that the extracted PDF text isn't noisy before chunking.

"Here is my Documents section, chunking strategy section, and pipeline mermaid diagram. Write a Python script that loads these sources by type — PyPDF for the PDFs, requests, and the UTEP pages, and reads the Reddit thread from a local .txt provided. Chunk : one entry per referral provider, one comment per Reddit thread, by section for the handouts. Then print a chunk so I can inspect."


**Milestone 4 — Embedding and retrieval:**

I will share my Retrieval Approach section (all-MiniLM-L6-v2, top-k = 3) and the chunk format. I expect it to suggest code that embeds each chunk with sentence-transformers, stores the vectors plus metadata in ChromaDB, and a retrieve()
function that returns the closest chunks to a query by distance score. I'll verify by
running 3  questions and checking the correct output for each. 




**Milestone 5 — Generation and interface:**
I will share my chunk format and grounding requirement. Must cite the real UTEP source link. I expect it to suggest something that sends the question plus the top chunks to Groq.I'll verify by running my evaluation questions and check each answer is accurate to the links for UTEP resources, and that asking something my documents don't cover makes the system say it doesn't have enough information instead of making something up. 
