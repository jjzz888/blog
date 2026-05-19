---
layout: post
title: "Agentic Memory"
date: 2026-05-19 00:00:00 +0800
categories: [llm, agents, memory]
---

Agentic memory refers to the ability of an AI agent to store, retrieve, and use information across time — both within a single task and across separate conversations or sessions.

Unlike a basic chatbot that only "knows" what's in the current conversation window, an agent with memory can build up a persistent understanding of the user, their work, and their preferences over time.

## The main types of agentic memory

**In-context memory** is the agent's short-term or working memory: everything currently inside the context window, such as the system prompt, recent conversation turns, tool results, and injected context. It is fast and immediately available, but it is temporary and limited by the model's context length.

**External / persistent memory** is information stored outside the model in files, databases, key-value stores, or vector stores. It is the agent's long-term memory layer: the agent writes useful facts, preferences, documents, and task state to it, then retrieves only the relevant pieces later. This is what allows an agent to remember your name, your preferences, ongoing project context, or past reference material across sessions.

**Semantic memory** refers to generalized knowledge: what the model already "knows" in its weights, such as facts, concepts, and patterns learned through pretraining or fine-tuning. It is different from external memory: semantic retrieval over embeddings is usually a way of searching external memory, not semantic memory itself.

**Episodic memory** captures past experiences as events: what the agent or user was trying to do, what actions were taken, what happened, and what was learned. It helps the agent reuse successful patterns, avoid repeating failures, and reason from prior experience rather than just from abstract facts.

**Procedural memory** encodes how to do things: skills, workflows, policies, or reusable task procedures the agent can follow or retrieve at execution time. It is less about remembering facts and more about remembering methods.

## Why it matters

Without memory, every interaction starts from scratch. With it, an agent can:

- Tailor responses to a specific user's role and expertise
- Avoid repeating mistakes the user has corrected before
- Maintain continuity across long-running projects
- Build up a richer model of the user's goals over time

The challenge is deciding *what* to remember, *when* to retrieve it, and how to keep memories from going stale as the world changes. These are active research and engineering problems in the field of AI agents.

## What is in the context window

The context window is the full "working memory" an LLM sees at inference time. The following is a conceptual prompt assembly order. The typical ordering, from top to bottom:

1. **System prompt** — Behavioral instructions first: the model's role, rules, and tone. Conceptually, this should stay lean and focused, with user/profile data treated as a separate layer even if some APIs serialize those layers together in a single top-level `system` field.

2. **User profile, durable preferences, project context** — Stable information about the user that shapes every response: who they are, their long-term preferences, active project requirements, key decisions. Extracted from the system prompt to keep instructions and data separate, and to make it easy to update dynamically each session.

3. **Tool definitions** — The structured schemas (name, description, parameters) that the model reads to know what tools it can call and how to call them. Typically injected by the platform/API automatically when tools are passed to the model.

4. **Conversation history** — The full back-and-forth in chronological order, which includes interleaved:
   - User messages
   - Assistant messages (including any tool calls the assistant made)
   - Tool results (the response returned after each tool call)

5. **Current user message** — The most recent input, sitting at the very end, with query-specific RAG context prepended inside it. Bundling retrieved evidence together with the query that triggered it takes advantage of the recency effect and keeps the two semantically linked.

A few nuances worth knowing: models tend to pay more attention to content at the very beginning and very end of the context (the "lost in the middle" phenomenon), which is why the system prompt and the current user message are so influential. As conversations grow long, older history gets truncated or summarized to make room — which is part of why persistent external memory matters.

## Example context window (Anthropic API format)

In Anthropic's Messages API, `system` and `tools` are top-level fields; retrieved context placement inside the final user turn is an application pattern, not a required API field.

```json
{
  "system": "You are a helpful customer support agent for Acme Store. Today's date is May 19, 2026. Be concise and friendly.",

  "tools": [
    {
      "name": "search_orders",
      "description": "looks up order status",
      "input_schema": {
        "type": "object",
        "properties": {
          "order_id": { "type": "string" }
        },
        "required": ["order_id"]
      }
    },
    {
      "name": "issue_refund",
      "description": "initiates a refund",
      "input_schema": {
        "type": "object",
        "properties": {
          "order_id": { "type": "string" },
          "amount": { "type": "number" }
        },
        "required": ["order_id", "amount"]
      }
    }
  ],

  "messages": [
    {
      "role": "user",
      "content": "Hi, I placed an order last week but haven't heard anything."
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help. Could you share your order number?"
    },
    {
      "role": "user",
      "content": "It's #A1042."
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "tu_001",
          "name": "search_orders",
          "input": { "order_id": "A1042" }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "tu_001",
          "content": "{\"status\": \"delayed\", \"estimated_delivery\": \"2026-05-22\", \"reason\": \"weather\"}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "Your order #A1042 is delayed due to weather and should arrive by May 22nd. Sorry for the inconvenience!"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Acme refund policy: customers may request a full refund if delivery is delayed more than 5 days past the original estimate. Original estimate for #A1042 was May 17."
        },
        {
          "type": "text",
          "text": "Can I get a refund?"
        }
      ]
    }
  ]
}
```

A few things to notice:

- **Tool results are sent as `role: user` messages**, not a special role. In normal tool-use flows, messages typically move between `user` and `assistant`, though Anthropic also supports special cases such as a final `assistant` prefill for constrained continuation.
- **The RAG context is prepended as a text block inside the final user message**, alongside the actual query, which is how most systems inject it in practice.
- **Tool calls appear as a content block inside an `assistant` message**, with a matching `tool_result` block in the next `user` message — they're always paired by `tool_use_id`.

## When the context window is full

When the context window is full, the agent needs to compress, discard, or retrieve instead of carrying the entire raw conversation forward. The usual fixes are:

- **Summarize old turns.** Replace earlier chat history with a compact summary of goals, decisions, constraints, and unresolved items.
- **Keep only working memory.** Preserve the most recent turns plus anything still actively relevant, and drop stale discussion.
- **Store long-term facts externally.** Move durable information like user preferences, project requirements, or important decisions into a database/vector store, then retrieve only what is needed later.
- **Log episodes separately.** Save important events such as "tried X, failed because Y" into an event log, instead of keeping every turn in the prompt.
- **Rebuild context per turn.** For each new request, construct a fresh prompt from, top to bottom:
   - **System instructions** — first, because they define the model's behavior, boundaries, and task framing.
   - **Stable user/profile memory** — next, because durable preferences, user facts, and project constraints should shape planning before the model decides how to act.
   - **Tool definitions** — after that, because the model needs to know what actions it can take once it is grounded in the user and task context.
   - **Summary of older turns** — this preserves continuity from the earlier conversation without replaying the full history.
   - **Retrieved facts/docs** — query-relevant external knowledge comes next so the model has the evidence it may need for the current request. If the task is knowledge-heavy, these usually matter more than past episodes, so they should be considered first.
   - **Relevant past episodes** — prior successes, failures, and lessons are included here when they can influence planning or tool use. These usually follow retrieved docs because they are experiential guidance rather than primary evidence, though for highly action-heavy workflows you might rank episodes above docs.
   - **Most recent turns** — these should sit close to the current request because they are the immediate local conversational context.
   - **Current user message** — last, so the model's attention lands on the exact request it must answer now. In implementation, query-specific retrieved documents are often bundled or prepended inside this final user turn; this list is a conceptual assembly order, not a required wire format.

Because of the lost-in-the-middle effect, content near the end of the prompt can receive more attention than content buried in the middle. In practice, that means the most important turn-specific item should often be placed close to the current user message. Which item deserves that placement depends on the task: if the key input is query-specific evidence, keep the retrieved documents right next to the final user turn; if the key input is immediate conversational continuity, keep the most recent turns there; if the key input is a prior failure or reusable lesson, surface the most relevant episode near the end as well. Stable instructions usually still belong near the top, because the beginning of the prompt is also highly salient. So the practical rule is not to keep one rigid order at all times, but to keep globally important rules near the top while moving turn-critical evidence and context closer to the current user message.

A useful pattern for this kind of dynamic ordering is:

- **System instructions**
- **Stable user/profile memory**
- **Tool definitions**
- **Summary of older turns**
- **Lower-priority retrieved/supporting context**
- **Recent turns**
- **Highest-priority retrieved doc or episode for this turn**
- **Current user message**

### Example in Anthropic Messages API format

The following example shows how the dynamic ordering pattern maps into Anthropic's actual Messages API wire format.

In Anthropic's API, that usually becomes:

- `system instructions`, `stable user/profile memory`, and `summary of older turns` inside the top-level `system` field
- `tool definitions` inside the top-level `tools` field
- `recent turns` inside `messages`
- `lower-priority retrieved/supporting context`, `highest-priority retrieved doc or episode for this turn`, and the `current user message` bundled into the final `user` turn

```json
{
  "model": "claude-sonnet-4-5",
  "max_tokens": 800,
  "system": "You are a helpful travel-planning agent. Follow company policy, respect saved user preferences, and avoid repeating known failures.\n\nStable user/profile memory:\n- User prefers aisle seats.\n- Trips should stay within a strict budget.\n- Team uses a shared approval process for exceptional bookings.\n\nSummary of older turns:\n- Goal: book compliant travel for the next trip.\n- Decisions: apply budget filters early and preserve booking-related failures as reusable lessons.\n- Constraints: validate traveler identity before purchase.\n- Open question: whether manager approval is needed for this itinerary.",
  "tools": [
    {
      "name": "search_flights",
      "description": "Find available flights that satisfy itinerary and policy constraints.",
      "input_schema": {
        "type": "object",
        "properties": {
          "origin": { "type": "string" },
          "destination": { "type": "string" },
          "date": { "type": "string" }
        },
        "required": ["origin", "destination", "date"]
      }
    },
    {
      "name": "book_flight",
      "description": "Book a selected itinerary after validation checks pass.",
      "input_schema": {
        "type": "object",
        "properties": {
          "itinerary_id": { "type": "string" },
          "traveler_id": { "type": "string" }
        },
        "required": ["itinerary_id", "traveler_id"]
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "Can you book my next trip to Tokyo for next Thursday?"
    },
    {
      "role": "assistant",
      "content": "Yes. I will use your saved preferences and check policy constraints before booking."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Lower-priority retrieved/supporting context:\n- Travel policy: bookings above the budget cap require manager approval.\n- Traveler profile docs: legal passport name must be validated before purchase."
        },
        {
          "type": "text",
          "text": "Highest-priority retrieved doc or episode for this turn:\n- Previous flight booking failed because the passport name did not match the legal document.\n  Lesson: validate the passport legal name before purchase."
        },
        {
          "type": "text",
          "text": "Please book the next trip to Tokyo for next Thursday, using my saved constraints."
        }
      ]
    }
  ]
}
```

## Concrete summarization schema

When you summarize older turns, a good structured summary keeps only the information that will matter later. A practical schema is:

- **`goal`**: the user's current objective
- **`decisions`**: decisions already made during the conversation
- **`constraints`**: requirements, limits, preferences, or non-negotiables
- **`durable_facts`**: stable facts worth carrying forward beyond the session
- **`open_questions`**: unresolved questions or missing information
- **`next_step`**: the next action the agent should take
- **`episode_notes`**: short notes about important failures, successes, or lessons learned

Example:

```json
{
  "goal": "Help the user build a travel assistant",
  "decisions": [
    "Use budget filtering early in the workflow",
    "Keep user seat preferences as durable memory"
  ],
  "constraints": [
    "Must support aisle seat preferences",
    "Must stay within a strict budget"
  ],
  "durable_facts": [
    "The team uses a shared approval process"
  ],
  "open_questions": [
    "How should episodic failures be stored and retrieved?"
  ],
  "next_step": "Design the episodic memory storage and retrieval flow",
  "episode_notes": [
    {
      "event": "Previous booking failed because the passport name did not match",
      "outcome": "failure"
    }
  ]
}
```

This kind of schema is useful because it separates long-lived facts from temporary working context, and it gives the agent a compact way to rebuild context without replaying the full conversation.


## External memory

External memory can be implemented with different storage types, and each one is good at holding different kinds of information.

### Types of storage

- **Structured storage** — relational databases such as Postgres or SQLite, or document databases with clear fields and indexes.
- **Vector store** — embedding-based storage for semantic similarity search, such as pgvector, Pinecone, Weaviate, FAISS, or Chroma.
- **Key-value store** — simple lookup storage such as Redis or a persistent KV database where data is fetched by a known key.
- **Files** — flat files or object storage such as JSON, JSONL, Markdown, CSV, Parquet, or logs in a filesystem or blob store.

### What to store in each type

- **Structured storage**
  - User profiles and preferences
  - Project metadata and task state
  - Episode records with fields like user, task type, timestamp, outcome, and lesson learned
  - Items that need filtering, sorting, joins, auditability, or updates

- **Vector store**
  - Embedded document chunks for semantic retrieval
  - Embedded summaries of episodes or memories for similarity search
  - Knowledge that should be retrieved by meaning rather than exact key or exact SQL filter
  - Usually not the source of truth by itself; often paired with structured storage or files

- **Key-value store**
  - Session state
  - Cached summaries or cached retrieval results
  - Small durable preferences when lookup is naturally by a stable key such as `user_id` or `session_id`
  - Fast-access memory that does not require complex querying

- **Files**
  - Append-only logs
  - JSON or JSONL event histories
  - Markdown notes and human-editable memory documents
  - Large offline datasets, exports, snapshots, and archives
  - Good for portability and inspection, but weaker than databases for high-concurrency querying and updates

### A practical rule of thumb

- Use **structured storage** when the memory has clear fields and you need reliable filtering or updates.
- Use a **vector store** when retrieval should be based on semantic similarity.
- Use a **key-value store** when the lookup key is already known and speed matters more than query flexibility.
- Use **files** when you want simple persistence, logging, portability, or human-readable records.

In practice, strong systems often combine them: for example, structured storage as the source of truth, a vector index for semantic lookup, a key-value layer for caching, and files for logs or offline archives.


## Episodic memory

Episodic memory stores past experiences: what the agent or user was trying to do, what actions were taken, what happened, and what was learned. It is especially useful for avoiding repeated failures, reusing successful workflows, and personalizing future actions based on prior outcomes.

### Where episodic memory comes from

Episodic memory comes from past interaction history, not just from raw conversation text. It is usually distilled from what the agent or user was trying to do, what actions were taken, what outcomes occurred, and what lessons emerged.

Common sources include:

- conversation history
- tool calls and tool results
- task outcomes
- errors, retries, and corrections
- workflow traces
- post-task summaries or reflections

Conversation history is one important source, but by itself it is often incomplete. The most useful episodic memories usually combine what was said with what was attempted and what actually happened.

### Where and how it is usually stored

Episodic memory is usually stored as structured event records, sometimes with a semantic index layered on top.

- **Structured database** — the most common source of truth for episodes.
  Example: a Postgres or SQLite table where each row stores fields such as `episode_id`, `user_id`, `task_type`, `timestamp`, `outcome`, `resolution`, and `lesson_learned`.

- **Document or event-log storage** — useful when you want append-only records or flexible JSON schemas.
  Example: a JSONL file where each line is one episode, or a document store where each episode is saved as a JSON object.

- **Vector store as a secondary index** — useful for finding semantically similar past episodes, but usually not ideal as the only source of truth.
  Example: embed a short textual summary such as "Flight booking failed because passport name mismatched; validate legal name before booking" and store that embedding in pgvector or Pinecone, linked back to the full structured record.

A practical pattern is to keep the full episode in structured storage, then optionally add an embedding index for semantic search.


A typical episodic-memory record might look like this:

```json
{
  "episode_id": "ep_1024",
  "user_id": "u_7",
  "session_id": "s_88",
  "task_type": "travel_booking",
  "timestamp": "2026-05-19T09:30:00Z",
  "goal": "Book a flight to Tokyo",
  "actions": [
    "searched flights",
    "selected itinerary",
    "submitted traveler details"
  ],
  "outcome": "failure",
  "error_reason": "passport name mismatch",
  "resolution": "use the exact legal passport name before purchase",
  "lesson_learned": "validate the passport legal name before booking",
  "tags": ["booking", "passport", "failure"]
}
```

### How to use episodic memory

Episodic memory should be used when prior outcomes can change the current plan, tool choice, or answer. Typical use cases include:

- repeated workflows
- user-specific recurring tasks
- debugging and troubleshooting
- tasks where previous failures or successful resolutions matter
- agent loops that need to learn from past actions

You generally do not retrieve episodic memory for every query. It is most useful when the current task is similar to something that happened before.

### Retrieval pipeline

A good episodic-memory retrieval pipeline is usually hybrid rather than purely semantic.

1. **Interpret the current request** — identify the task type, entities, user, constraints, and whether past experience is likely to matter.
2. **Filter candidate episodes** — narrow the search space by metadata such as `user_id`, `task_type`, recency, project, or success/failure status.
3. **Optionally run semantic search** — retrieve semantically similar episodes using embeddings over compact episode summaries.
4. **Rank the candidates** — rank by a mix of semantic similarity, metadata match, recency, and actionability of the lesson.
5. **Select only the best few** — inject a small number of relevant episodes, not the entire log.
6. **Convert them into compact prompt text** — include only the parts that help the current decision, such as the failure cause and the lesson.

In other words, the current user query often drives retrieval, but the query should usually be interpreted together with recent turns, conversation summary, and task metadata.

### What the injected memory looks like

Injected episodic memory should usually be compact and action-oriented. Instead of dumping raw JSON into the prompt, summarize the episode into a short lesson or warning.

Example of a compact injected memory block:

```text
Relevant past episode:
- Previous flight booking failed because the passport name did not match the legal document.
  Lesson: validate the passport legal name before purchase.
```

### Example in Anthropic Messages API format

In Anthropic's Messages API, query-scoped episodic memory is often bundled into the final user turn so it stays adjacent to the current request.

```json
{
  "system": "You are a helpful travel-planning agent. Follow policy, respect saved preferences, and avoid repeating known failures.",
  "tools": [
    {
      "name": "search_flights",
      "description": "Find available flights that satisfy itinerary and policy constraints.",
      "input_schema": {
        "type": "object",
        "properties": {
          "origin": { "type": "string" },
          "destination": { "type": "string" },
          "date": { "type": "string" }
        },
        "required": ["origin", "destination", "date"]
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "Can you remember that our team uses a shared approval process?"
    },
    {
      "role": "assistant",
      "content": "Yes. I will keep that in mind for future bookings."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Relevant past episode:\n- Previous flight booking failed because the passport name did not match the legal document. Lesson: validate the passport legal name before purchase."
        },
        {
          "type": "text",
          "text": "Please book the next trip with the saved constraints in mind."
        }
      ]
    }
  ]
}
```

This example shows the usual pattern: the structured episode lives outside the prompt in persistent storage, but a short, query-relevant summary of that episode is injected right next to the current user message.


## Agent memory flow

A cleaner way to describe the full memory loop is the following:

1. **User sends a message**
   A new request arrives.

2. **Interpret the request**
   The system identifies:
   - the task type
   - the user
   - key entities and constraints
   - whether memory retrieval is needed

3. **Retrieve relevant memory and knowledge**
   The system gathers only the most useful context for this turn, which may include:
   - **Stable user/profile memory** from structured or key-value storage
   - **Relevant external knowledge** from vector stores, SQL databases, files, or document stores
   - **Relevant episodic memory** from past experiences, retrieved by metadata filters, recency, and semantic similarity
   - **Recent conversation history** and, if needed, a summary of older turns

4. **Assemble the context**
   The system constructs a prompt with an intentional order, for example:
   - system instructions
   - stable user/profile memory
   - tool definitions
   - summary of older turns
   - lower-priority retrieved/supporting context
   - recent turns
   - highest-priority retrieved doc or episode for this turn
   - current user message

5. **LLM reasons over the assembled context**
   The model decides whether it can answer directly or needs tools.

6. **Tool-use loop (if needed)**
   The model may call tools such as:
   - search
   - code runner
   - APIs

   Tool results are added back into context, and the model may reason again or call more tools before producing the final answer.

7. **Selectively write memory**
   After or during the interaction, the system decides what is worth saving:
   - **Save an episode** if a meaningful task, failure, success, or lesson occurred
   - **Update stable profile memory** only for durable, high-confidence user facts or preferences
   - **Store retrieval-worthy outputs** by summarizing and embedding useful documents, conclusions, or traces
   - **Summarize/compress history** when needed for future context management

   In practice, the system often decides what is worth saving after each turn or meaningful step. It may also summarize or selectively write memory in the middle of a turn when context pressure becomes high or when an important event, such as a failure or correction, should be preserved immediately. This means long-term memory writing and short-term context compression are related but not identical: one is mainly about durable retention, while the other is mainly about keeping the current interaction within the context window.

8. **Deliver the response**
   The final answer is returned to the user.
