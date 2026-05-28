---
layout: post
title: "Implementation of Agentic Memory"
date: 2026-05-28 00:00:00 +0800
categories: [llm, agents, memory]
---

Source: agentic-memory

This note turns the conceptual article into an implementation checklist and API design. The core runtime loop is:

1. Receive a user turn.
2. Interpret the task and decide what context is needed.
3. Retrieve relevant profile memory, external knowledge, episodic memory, procedural memory, MCP resources, conversation history, and summaries.
4. Assemble a provider-specific context window with intentional ordering.
5. Let the model answer directly or enter a capability-use loop.
6. Selectively write durable memories, episode records, summaries, caches, and audit events.

## 1. Domain Model

The domain model defines the core concepts, relationships, and invariants of agentic memory. It is independent of programming language, network protocol, storage backend, and LLM provider.

The domain model answers: **what are the important things in the system, and how do they relate?**

### 1.1. Core concepts

Separate stable domain entities from runtime artifacts. Domain entities are the things the memory system stores or reasons about. Runtime artifacts are how the agent executes a turn, talks to model providers, and records observability.

Domain entities:

| Object | Meaning |
|---|---|
| `ActorRef` | Tenant, user, project, and session identity used for scoping memory and access control. |
| `Turn` | One agent interaction from user input through final response, including state transitions and capability calls. |
| `Message` | Provider-neutral user, assistant, system, or tool message. |
| `ConversationSummary` | Compressed older context with goal, decisions, constraints, durable facts, open questions, next step, and episode notes. |
| `ProfileMemory` | Durable user, project, tenant, or session facts and preferences. |
| `KnowledgeItem` | External document, chunk, database result, file reference, or retrieval-worthy output. |
| `Episode` | Structured record of a past task, approach, actions, outcome, error, resolution, cost, quality, and lesson. |
| `Procedure` | Skill, playbook, policy, checklist, workflow, or guide that tells the agent how to act. |
| `Capability` | A tool, procedure/skill, external resource provider, resource template, or provider-native capability. |
| `ExternalCapabilityProvider` | External integration that can expose tools, resources, prompts, or templates. MCP is one implementation of this concept. |
| `ExternalResource` | Loaded external resource from an external provider such as MCP, files, databases, or service APIs. |
| `PrivacyPolicy` | Tenant/user/project-scoped privacy, consent, sensitivity, retention, and forget/export policy for stored memories. |

Runtime artifacts:

| Object | Meaning |
|---|---|
| `TurnState` | Enum of persisted executor state values. |
| `TurnMachineContext` | Persisted execution context for a turn, including current state, iteration count, pending capability calls, errors, and limits. |
| `StateTransitionRecord` | Durable record of a state-machine transition with reason and metadata. |
| `ToolDefinition` | Provider-neutral callable action schema. |
| `ToolCall` | Provider-neutral request to execute a callable action. |
| `ToolResult` | Provider-neutral result from executing a callable action. |
| `RetrievedContextItem` | A ranked piece of context selected for possible injection into a model request. |
| `ContextAssembly` | Ordered, token-budgeted context ready for provider rendering. |
| `ModelRequest` | Provider-neutral request to an LLM provider. |
| `ModelResponse` | Provider-neutral model output, including text, tool requests, usage, stop reason, and raw payload reference. |
| `HookEvent` | Typed event emitted before or after retrieval, context assembly, model calls, tools, skills, MCP, memory writes, and errors. |
| `AuditEvent` | Durable record of decisions, retrievals, state transitions, capability calls, and memory writes. |

### 1.2. Concept relationships

- A `Turn` contains messages, state transitions, retrieved context, capability calls, tool results, hook events, memory-write candidates, and final output.
- A `TurnMachineContext` is the primary persisted execution aggregate for a `Turn`; `TurnRecord` is a read projection over that context and the transition log.
- A `TurnState` is the enum value inside `TurnMachineContext`; it does not include iteration counters, pending tool calls, or errors by itself.
- A `StateTransitionRecord` belongs to a `Turn` and makes state-machine behavior replayable, auditable, and resumable.
- A `ConversationSummary` compresses older `Message` objects while preserving goals, decisions, constraints, durable facts, open questions, and next steps.
- A `ProfileMemory` stores durable facts or preferences about an actor, project, tenant, or session.
- An `Episode` is usually derived from a completed or failed `Turn`, then used as future experiential guidance.
- A `KnowledgeItem` represents external evidence or source material, while an `Episode` represents past experience.
- A `Procedure` describes how to act; it can be surfaced as a skill, policy, playbook, checklist, workflow, or external-resource guide.
- A `Capability` is something the agent can use, such as a tool, skill, procedure, external resource provider, resource template, or provider-native feature.
- A `ContextAssembly` contains selected and ordered `RetrievedContextItem` objects, plus instructions, recent turns, tool definitions, and capability context.
- A `ModelRequest` is rendered from a `ContextAssembly` and selected capabilities.
- A `ModelResponse` may produce final text or neutral `ToolCall` objects.
- A `ToolResult` is appended back into the turn and may cause the model to continue.
- A `HookEvent` observes or controls runtime behavior around retrieval, model calls, tool calls, MCP calls, skill loading, memory writes, errors, and finalization.
- An `AuditEvent` records why retrieval, state transitions, capability calls, and memory writes happened.
- `PrivacyPolicy` applies to repository queries and writes so tenant scope, consent, sensitivity classification, retention, export, and forgetting are enforced below the service layer.

### 1.3. Provider-neutral domain concepts

The domain model should use neutral concepts even when providers represent them differently. Provider-specific shapes belong in adapters.

Examples:

```text
Domain concept: ToolCall
Anthropic mapping: tool_use content block
OpenAI mapping: tool_calls entry
Local model mapping: structured JSON call
```

```text
Domain concept: Procedure packaged as Skill
Anthropic mapping: container.skills when supported
Fallback mapping: loaded procedure text in the capability registry
```

```text
Domain concept: McpServer
Anthropic mapping: mcp_servers and mcp_toolset when supported
Fallback mapping: local MCP client wrapped as tools/resources
```

### 1.4. Implementation scope by stage

The domain model is broader than the first implementation. Keep all concepts in the design, but implement them by stage based on how directly they support the memory loop. `Phase 1` is the first working implementation; `Phase 2` and `Phase 3` are post-Phase 1 expansion stages.

| Concept | Relationship to agentic memory | Implementation stage | Python data contract | Suggested package/module |
|---|---|---|---|---|
| `ActorRef` | Scopes memory by tenant, user, project, and session so retrieval and writes do not leak across boundaries. | Phase 1 | 8.1 `ActorRef` | `schemas/actors.py` |
| `Turn` | Main unit where memory is retrieved, injected, used, and written back. | Phase 1 | 8.14 `TurnRecord` | `schemas/turns.py`; `repositories/turns.py` |
| `TurnState` | Names the current executor state in the turn state machine. | Phase 1 | 8.13 `TurnState` | `schemas/turns.py`; `services/turn_state_machine.py` |
| `TurnMachineContext` | Primary persisted execution aggregate for resumable turn execution. | Phase 1 | 8.13 `TurnMachineContext` | `schemas/turns.py`; `services/turn_state_machine.py`; `repositories/turns.py` |
| `StateTransitionRecord` | Durable state-machine transition log used for audit, replay, and recovery. | Phase 1 | 8.14 `StateTransitionRecord` | `schemas/audit.py`; `services/audit_logger.py`; `repositories/audit.py` |
| `Message` | Source material for conversation memory, summaries, episodes, and memory-write candidates. | Phase 1 | 8.2 `ConversationMessage` | `schemas/messages.py`; `repositories/messages.py` |
| `ConversationSummary` | Compresses long conversation history into reusable memory when raw messages no longer fit. | Phase 2 | 8.3 `ConversationSummary`, `EpisodeNote` | `schemas/summaries.py`; `services/summarizer.py`; `repositories/summaries.py` |
| `ProfileMemory` | Stores durable user, project, tenant, or session facts and preferences. | Phase 1 | 8.4 `ProfileMemory` | `schemas/memory.py`; `repositories/profile_memory.py` |
| `KnowledgeItem` | Stores external evidence, documents, files, chunks, or generated outputs for retrieval. | Phase 1 | 8.5 `KnowledgeDocument`, `KnowledgeChunk` | `schemas/knowledge.py`; `repositories/knowledge.py` |
| `Episode` | Stores past task outcomes and lessons for future experiential recall. | Phase 1, with a simple schema | 8.7 `EpisodeRecord` | `schemas/episodes.py`; `repositories/episodes.py` |
| `Procedure` | Stores reusable ways of working as procedural memory. | Phase 1, as local records | 8.8 `ProcedureRecord` | `schemas/procedures.py`; `repositories/procedures.py` |
| `Capability` | Generalizes tools, procedures, MCP, and provider-native features that the agent can use. | Phase 2 for local capability abstraction; Phase 3 for MCP and provider-native implementations | 8.8 `CapabilityRegistryEntry`, `ToolDefinition`, `McpServerRecord`, `McpResource`, `McpResourceTemplate` | `schemas/capabilities.py`; `services/capability_registry.py`; `services/capability_loop.py` |
| `ToolDefinition` | Describes model-visible callable actions. | Phase 2 | 8.8 `ToolDefinition` | `schemas/tools.py`; `services/tool_runner.py` |
| `ToolCall` | Captures model-requested actions that can later become episode evidence or memory-write inputs. | Phase 2 | 8.11 `ToolCall` | `schemas/tools.py`; `services/tool_runner.py`; `integrations/anthropic/tool_loop.py` |
| `ToolResult` | Captures action outputs that can become knowledge, episode, or memory-write candidates. | Phase 2 | 8.11 `ToolResult` | `schemas/tools.py`; `services/tool_runner.py`; `integrations/anthropic/tool_loop.py` |
| `McpServer` | Connects external memory-adjacent systems such as files, tickets, databases, or knowledge services. | Phase 3 | 8.8 `McpServerRecord`; 8.12 `AnthropicMcpServerConfig` for Anthropic adapter config | `schemas/mcp.py`; `services/mcp_manager.py`; `repositories/mcp.py`; `integrations/mcp/client.py` |
| `McpResource` | Represents external context loaded from MCP or an equivalent integration. | Phase 3 | 8.8 `McpResource`, `McpResourceTemplate` | `schemas/mcp.py`; `services/mcp_manager.py`; `integrations/mcp/resources.py` |
| `PrivacyPolicy` | Enforces sensitivity classification, consent, retention, export, forgetting, and tenant scoping for memory records. | Phase 1 for tenant scoping and basic sensitivity; Phase 2 for full consent/retention workflows | 8.1 `PrivacyClassification`, `RetentionPolicy`, `ConsentRecord`, `PrivacyPolicy` | `schemas/privacy.py`; `services/privacy.py`; `repositories/*` |
| `RetrievedContextItem` | Normalized retrieved memory candidate before final context injection. | Phase 1 | 8.9 `RetrievedContextItem` | `schemas/context.py`; `services/retrieval.py` |
| `ContextAssembly` | Ordered provider-neutral context that determines which memories the model sees. | Phase 1 | 8.14 `ContextAssembly` | `schemas/context.py`; `services/context_manager.py` |
| `ModelRequest` | Provider-neutral model input after memory context has been assembled. | Phase 1 | 8.11 `ModelRequest` | `schemas/model.py`; `llm/renderer.py`; `services/model_gateway.py` |
| `ModelResponse` | Model output used to produce final answers and candidate memory writes. | Phase 1 | 8.11 `ModelResponse` | `schemas/model.py`; `llm/base.py`; `services/model_gateway.py` |
| `HookEvent` | Extension point for memory retrieval and memory write boundaries. | Phase 1, memory-specific hooks only | 8.11 `HookEvent`, `HookDefinition`, `HookResult` | `schemas/hooks.py`; `services/hook_manager.py` |
| `AuditEvent` | Durable explanation of retrieval, injection, state transitions, and memory writes. | Phase 1 | 8.14 `AuditEvent` | `schemas/audit.py`; `services/audit_logger.py`; `repositories/audit.py` |

Phase 1 should implement the first complete memory baseline with the persisted state machine: receive a turn, interpret it, retrieve profile, simple episode, procedure, and knowledge/vector memory, assemble context, call the model once, write selected memory, and audit the path. To keep Phase 1 practical, request interpretation, ranking, and importance scoring should start as deterministic or lightweight heuristics; LLM-based interpretation/scoring is a Phase 2 quality upgrade. Phase 2 adds post-Phase 1 retrieval quality, summaries, local tools, richer procedures, and lifecycle management. Phase 3 adds MCP, provider-native capabilities, distributed transports, and other external integrations.

## 2. Logical API Design

The logical API defines the operations that can be performed on the domain model. It is independent of HTTP, WebSocket, gRPC, queues, local function calls, Python classes, TypeScript interfaces, SQL schemas, Anthropic payloads, OpenAI payloads, and MCP wire details.

The logical API answers: **what can the system do with the domain model?**

Design rules:

- Use provider-neutral concepts such as `ModelRequest`, `ModelResponse`, `ToolCall`, `ToolResult`, `Procedure`/`Skill`, `McpResource`, `ContextAssembly`, and `TurnState`.
- Keep provider-specific shapes out of the logical API. Anthropic `tool_use`, Anthropic `tool_result`, Anthropic `container.skills`, Anthropic `mcp_toolset`, OpenAI `tool_calls`, and local-model JSON calling conventions belong in adapters.
- Keep protocol-specific shapes out of the logical API. `POST /agent/turns`, WebSocket events, gRPC methods, queue messages, and stdio messages are transport mappings.
- Keep implementation-specific shapes out of the logical API. Pydantic, TypeScript interfaces, SQLAlchemy models, and database schemas are implementation mappings.

### 2.1. Logical operations

| Operation                  | Purpose                                                                                                                                           | Provider/protocol notes                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RunTurn`                  | Execute a complete memory-aware turn from user input to final answer.                                                                             | May map to HTTP, WebSocket, local function, or queue-backed workflow.                                                                                         |
| `ResumeTurn`               | Resume a non-terminal turn after crash recovery or retryable interruption without adding new model-visible input.                                 | Uses persisted `TurnMachineContext`; terminal `completed`, `failed`, and `cancelled` turns should be queried or superseded by a new turn rather than resumed. |
| `SubmitToolResults`        | Continue a turn after model-requested tool calls have completed.                                                                                  | Maps neutral `ToolResult` records back into the selected provider format.                                                                                     |
| `SubmitApproval`           | Continue or stop a turn after a human approval decision.                                                                                          | Useful for approval-gated tools and memory writes.                                                                                                            |
| `GetTurnState`             | Read the current state, iteration count, pending tool calls, partial outputs, and errors for a turn.                                              | Useful for UI polling, WebSocket resume, and audit views.                                                                                                     |
| `TransitionTurnState`      | Move a turn through the state machine with an explicit reason.                                                                                    | Must emit an `AuditEvent`.                                                                                                                                    |
| `InterpretRequest`         | Classify task type, entities, constraints, retrieval needs, and likely capability needs.                                                          | Provider-neutral; may use an LLM internally.                                                                                                                  |
| `EstimateContextBudget`    | Estimate token budget and reserve space for instructions, tools, retrieved context, history, and output.                                          | Provider adapters provide model-specific limits.                                                                                                              |
| `IsCompressionNeeded`      | Decide whether older context should be summarized, dropped, or externally stored.                                                                 | Independent of transport.                                                                                                                                     |
| `SummarizeConversation`    | Create or update structured conversation summaries.                                                                                               | May run synchronously or as a background job.                                                                                                                 |
| `GetConversationHistory`   | Retrieve recent messages and tool results for a session.                                                                                          | Transport mapping can paginate or stream.                                                                                                                     |
| `SearchProfileMemory`      | Retrieve durable user/project facts and preferences.                                                                                              | Usually exact or hybrid structured search.                                                                                                                    |
| `PatchProfileMemory`       | Add, update, supersede, or delete durable profile memories.                                                                                       | Must handle conflicts and provenance.                                                                                                                         |
| `ForgetByActor`            | Delete or tombstone memories for an actor according to privacy and retention policy.                                                              | Must enforce tenant scoping and audit the action.                                                                                                             |
| `ExportByActor`            | Export actor-scoped memory records for inspection or portability.                                                                                 | Must apply privacy policy and redact records that cannot be exported.                                                                                         |
| `UpsertKnowledge`          | Store external documents, chunks, files, database results, or generated outputs.                                                                  | Source-of-truth storage should be separate from vector indexes.                                                                                               |
| `SearchKnowledge`          | Retrieve external knowledge by metadata, keyword, vector search, graph traversal, or hybrid search.                                               | Retrieval strategy is backend-specific; result shape is neutral.                                                                                              |
| `UpsertVectorIndex`        | Add or update an embedding entry for a memory, episode, document chunk, or procedure.                                                             | Backend adapters normalize relevance scores.                                                                                                                  |
| `SearchVectorIndex`        | Perform semantic search over indexed items.                                                                                                       | May map to ChromaDB, pgvector, Qdrant, Pinecone, or local indexes.                                                                                            |
| `CreateEpisode`            | Store a structured past task outcome and searchable text representation.                                                                          | Should include approach, outcome, cost, quality, error, and lesson.                                                                                           |
| `SearchEpisodes`           | Retrieve relevant prior experiences for the current task.                                                                                         | Usually hybrid: metadata filter, vector search, recency, and actionability.                                                                                   |
| `RenderEpisodes`           | Convert selected episodes into compact prompt-ready lessons.                                                                                      | Provider-neutral text/context block.                                                                                                                          |
| `SearchProcedures`         | Find relevant skills, policies, playbooks, workflows, and checklists.                                                                             | Can retrieve local skills or provider-native capabilities.                                                                                                    |
| `LoadSkill`                | Load the full procedure or instruction pack for a selected skill.                                                                                 | Adapter may render as Anthropic Skills or fallback context.                                                                                                   |
| `ListCapabilities`         | Return available tools, skills, MCP servers, resources, templates, and provider-native capabilities.                                              | Used to build capability registry.                                                                                                                            |
| `RenderCapabilityRegistry` | Build compact capability context for the model.                                                                                                   | Provider adapter decides whether capabilities are native or prompt-side.                                                                                      |
| `RenderToolsForProvider`   | Convert neutral `ToolDefinition` objects into provider-specific tool schemas.                                                                     | Anthropic/OpenAI/local adapters differ here.                                                                                                                  |
| `ExecuteToolCall`          | Execute a neutral `ToolCall` and return a neutral `ToolResult`.                                                                                   | Tool execution stays outside provider adapters.                                                                                                               |
| `ListMcpServers`           | List available MCP integrations and their capabilities.                                                                                           | May be native provider MCP or local MCP client.                                                                                                               |
| `ReadMcpResource`          | Load an MCP resource or resource-template result as a context item.                                                                               | Provider-neutral result; Anthropic MCP connector is an adapter detail.                                                                                        |
| `RenderMcpForProvider`     | Convert neutral MCP config into provider-specific config or fallback tools.                                                                       | Anthropic may use `mcp_servers` and `mcp_toolset`; others may use tool wrappers.                                                                              |
| `AssembleContext`          | Build ordered context from instructions, profile memory, summaries, retrieved knowledge, episodes, procedures, recent turns, and current message. | Provider-neutral assembly.                                                                                                                                    |
| `RenderModelRequest`       | Convert `ContextAssembly`, tools, skills, MCP config, and options into a provider payload.                                                        | Provider-specific adapter boundary.                                                                                                                           |
| `CallModel`                | Send a `ModelRequest` to a selected provider and return a normalized `ModelResponse`.                                                             | Anthropic/OpenAI/local implementations differ.                                                                                                                |
| `SubmitToolResults`        | Append tool results and continue model reasoning.                                                                                                 | Provider adapter maps neutral `ToolResult` to provider-specific wire format.                                                                                  |
| `ScoreMemoryImportance`    | Score whether a candidate memory is worth saving.                                                                                                 | May use rules, a small LLM, or both.                                                                                                                          |
| `ProposeMemoryWrites`      | Generate candidate profile, episode, summary, knowledge, or cache writes after/during a turn.                                                     | Should include reason, confidence, provenance, and review mode.                                                                                               |
| `CommitMemoryWrites`       | Persist approved memory writes and update indexes.                                                                                                | Should be idempotent.                                                                                                                                         |
| `RankMemory`               | Rank candidate context by relevance, importance, recency, metadata match, and actionability.                                                      | Backend-agnostic scoring policy.                                                                                                                              |
| `CheckMemoryConflicts`     | Detect contradictions, stale facts, duplicates, and supersession candidates.                                                                      | Important for long-lived profile/project memory.                                                                                                              |
| `ConsolidateMemory`        | Merge near-duplicate or overlapping memories into canonical summaries.                                                                            | Should supersede old records rather than deleting source records.                                                                                             |
| `DispatchHook`             | Run typed hook handlers around runtime events.                                                                                                    | Can map to Python handlers, Claude Code hooks, webhooks, or event bus messages.                                                                               |
| `WriteAuditEvent`          | Persist retrieval decisions, injected context, state transitions, capability calls, errors, and memory writes.                                    | Required for debugging and governance.                                                                                                                        |
| `RunEvaluation`            | Evaluate retrieval quality, ranking, context assembly, memory writes, and state-machine behavior.                                                 | Can run offline or in CI.                                                                                                                                     |

### 2.2. Provider-neutral capability mapping

| Neutral concept         | Anthropic mapping                                                           | OpenAI/local mapping                                             |
| ----------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `ToolDefinition`        | Messages API `tools` entry.                                                 | OpenAI tool schema or local function registry.                   |
| `ToolCall`              | `tool_use` content block.                                                   | OpenAI `tool_calls` entry or local structured call.              |
| `ToolResult`            | `tool_result` content block in the next user message.                       | OpenAI tool output item or local structured message.             |
| `Procedure`/`Skill`     | Anthropic `container.skills` when supported; otherwise prompt-side context. | Prompt-side procedure context or provider-specific skill system. |
| `McpServer`             | Anthropic `mcp_servers` when supported.                                     | Local MCP client wrapped as tools/resources.                     |
| MCP capability exposure | Anthropic `mcp_toolset` tool entry when supported.                          | Tool wrapper around local MCP calls.                             |
| `ContextAssembly`       | `system`, `messages`, `tools`, and optional provider config.                | Provider-specific messages/request format.                       |
| `HookEvent`             | Python hook bus or Claude Code hook integration.                            | Python hook bus, webhook, event bus, or local callback.          |

### 2.3. Transport and implementation mapping principle

Treat every concrete interface as a mapping from the logical API:

- HTTP endpoints are one mapping.
- WebSocket and SSE events are streaming mappings.
- gRPC methods are internal service mappings.
- Queues and event buses are durable async mappings.
- Local function calls are in-process mappings.
- MCP is a tool/resource integration mapping.
- Pydantic, SQLAlchemy, TypeScript, and database schemas are implementation mappings.

The logical API should remain stable even if a transport, LLM provider, storage engine, or implementation language changes.

### 2.4. Implementation scope by stage

The logical API is a roadmap, not a requirement that every operation must exist in the first version. Implement the operations by stage according to how directly they support the memory loop. `Phase 1` is the first working implementation; `Phase 2` and `Phase 3` are post-Phase 1 expansion stages.

| Operation                  | Relationship to agentic memory                                                                                                                                                   | Implementation stage                                                             | Python data contract                                                                                                   | Suggested package/module                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `RunTurn`                  | Executes the full memory path from user input to final response and memory write.                                                                                                | Phase 1                                                                          | 8.13 `TurnMachineContext`; 8.14 `TurnRecord`                                                                           | `services/turn_runner.py`; `services/turn_state_machine.py`; `repositories/turns.py`           |
| `ResumeTurn`               | Resumes a persisted non-terminal turn after retryable interruption or crash without adding new model-visible input. Cancelled turns are terminal and should not resume in place. | Phase 1                                                                          | 8.13 `TurnMachineContext`; 8.14 `TurnRecord`, `StateTransitionRecord`; 8.16 `ResumeTurnRequest`                        | `services/turn_runner.py`; `services/turn_state_machine.py`; `repositories/turns.py`           |
| `SubmitToolResults`        | Feeds completed tool results back into the capability loop.                                                                                                                      | Phase 2                                                                          | 8.11 `ToolResult`; 8.13 `TurnMachineContext`; 8.16 `SubmitToolResultsRequest`                                          | `services/capability_loop.py`; `services/tool_runner.py`                                       |
| `SubmitApproval`           | Applies a human approval or denial decision to a waiting turn.                                                                                                                   | Phase 2                                                                          | 8.13 `TurnMachineContext`; 8.16 `SubmitApprovalRequest`                                                                | `services/capability_loop.py`; `services/turn_state_machine.py`                                |
| `GetTurnState`             | Reads the current memory-path state for audit, debugging, and resumability.                                                                                                      | Phase 1                                                                          | 8.13 `TurnMachineContext`; 8.14 `TurnRecord`                                                                           | `services/turn_state_machine.py`; `repositories/turns.py`                                      |
| `TransitionTurnState`      | Records explicit lifecycle movement through interpretation, retrieval, context assembly, model call, and memory write.                                                           | Phase 1                                                                          | 8.13 `TurnState`; 8.14 `StateTransitionRecord`                                                                         | `services/turn_state_machine.py`; `services/audit_logger.py`; `repositories/turns.py`          |
| `InterpretRequest`         | Determines task type, entities, constraints, and memory retrieval needs.                                                                                                         | Phase 1                                                                          | 8.14 `InterpretedRequest`, `RetrievalNeed`                                                                             | `schemas/interpretation.py`; `services/interpreter.py`                                         |
| `EstimateContextBudget`    | Prevents retrieved memory from exceeding the model context budget.                                                                                                               | Phase 1, simple estimate                                                         | 8.11 `ProviderOptions`; 8.14 `RetrievalPlan`, `ContextAssembly`                                                        | `services/context_manager.py`; `services/model_gateway.py`; `schemas/providers.py`             |
| `IsCompressionNeeded`      | Decides when old context should become summaries instead of raw messages.                                                                                                        | Phase 2                                                                          | 8.2 `ConversationMessage`; 8.3 `ConversationSummary`; 8.14 `ContextAssembly`                                           | `services/context_manager.py`; `services/summarizer.py`                                        |
| `SummarizeConversation`    | Produces durable compressed conversation memory for long sessions.                                                                                                               | Phase 2                                                                          | 8.3 `ConversationSummary`, `EpisodeNote`                                                                               | `services/summarizer.py`; `repositories/summaries.py`; `workers/summarize.py`                  |
| `GetConversationHistory`   | Retrieves recent messages used as short-term conversational memory.                                                                                                              | Phase 1                                                                          | 8.2 `ConversationMessage`; 8.3 `ConversationSummary`                                                                   | `repositories/messages.py`; `repositories/sessions.py`                                         |
| `SearchProfileMemory`      | Retrieves durable facts and preferences.                                                                                                                                         | Phase 1                                                                          | 8.4 `ProfileMemory`; 8.9 `RetrievedContextItem`                                                                        | `services/retrieval.py`; `repositories/profile_memory.py`                                      |
| `PatchProfileMemory`       | Writes or updates durable facts and preferences.                                                                                                                                 | Phase 1                                                                          | 8.4 `ProfileMemory`; 8.14 `MemoryWriteCandidate`, `CommittedMemoryWrite`                                               | `services/memory_writer.py`; `repositories/profile_memory.py`; `repositories/memory_writes.py` |
| `ForgetByActor`            | Applies user/project forget requests and retention-policy deletes.                                                                                                               | Phase 2                                                                          | 8.1 `PrivacyPolicy`, `RetentionPolicy`; 8.14 `AuditEvent`                                                              | `services/privacy.py`; `repositories/*`; `services/audit_logger.py`                            |
| `ExportByActor`            | Exports actor-scoped memory records for inspection or portability.                                                                                                               | Phase 2                                                                          | 8.1 `PrivacyPolicy`; memory record contracts                                                                           | `services/privacy.py`; `repositories/*`                                                        |
| `UpsertKnowledge`          | Stores external documents, chunks, files, or generated outputs as retrievable memory.                                                                                            | Phase 1                                                                          | 8.5 `KnowledgeDocument`, `KnowledgeChunk`; 8.6 `VectorUpsertRequest`, `VectorUpsertResult`                             | `repositories/knowledge.py`; `repositories/vector_index.py`; `services/memory_writer.py`       |
| `SearchKnowledge`          | Retrieves external evidence or source material.                                                                                                                                  | Phase 1                                                                          | 8.5 `KnowledgeDocument`, `KnowledgeChunk`; 8.6 `VectorSearchRequest`, `VectorSearchResult`; 8.9 `RetrievedContextItem` | `services/retrieval.py`; `repositories/knowledge.py`; `repositories/vector_index.py`           |
| `UpsertVectorIndex`        | Maintains semantic indexes for memories, episodes, documents, and procedures.                                                                                                    | Phase 1                                                                          | 8.6 `VectorUpsertRequest`, `VectorUpsertResult`                                                                        | `repositories/vector_index.py`; `embeddings/base.py`; `workers/embed.py`                       |
| `SearchVectorIndex`        | Enables semantic retrieval beyond exact metadata or keyword search.                                                                                                              | Phase 1                                                                          | 8.6 `VectorSearchRequest`, `VectorSearchResult`                                                                        | `services/retrieval.py`; `repositories/vector_index.py`                                        |
| `CreateEpisode`            | Stores task outcomes and lessons as experiential memory.                                                                                                                         | Phase 1, simple schema                                                           | 8.7 `EpisodeRecord`                                                                                                    | `services/memory_writer.py`; `repositories/episodes.py`                                        |
| `SearchEpisodes`           | Retrieves relevant past task experiences.                                                                                                                                        | Phase 1, simple search                                                           | 8.7 `EpisodeRecord`; 8.9 `RetrievedContextItem`                                                                        | `services/retrieval.py`; `repositories/episodes.py`                                            |
| `RenderEpisodes`           | Converts selected episodes into compact retrieved context.                                                                                                                       | Phase 1                                                                          | 8.7 `EpisodeRecord`; 8.9 `RetrievedContextItem`                                                                        | `services/retrieval.py`; optional `services/episode_renderer.py`                               |
| `SearchProcedures`         | Retrieves reusable local procedures, playbooks, policies, and checklists.                                                                                                        | Phase 1                                                                          | 8.8 `ProcedureRecord`; 8.9 `RetrievedContextItem`                                                                      | `services/retrieval.py`; `repositories/procedures.py`                                          |
| `LoadSkill`                | Loads a full packaged skill or richer procedure content.                                                                                                                         | Phase 2                                                                          | 8.8 `ProcedureRecord`, `CapabilityRegistryEntry`; 8.12 `AnthropicSkillConfig` for Anthropic adapter rendering          | `services/skill_manager.py`; `integrations/anthropic/skills.py`                                |
| `ListCapabilities`         | Lists tools, skills, MCP, and provider-native capabilities for model-visible use.                                                                                                | Phase 2 for local tools/skills; Phase 3 for MCP and provider-native capabilities | 8.8 `CapabilityRegistryEntry`, `ToolDefinition`, `McpServerRecord`, `McpResourceTemplate`                              | `services/capability_registry.py`; `schemas/capabilities.py`                                   |
| `RenderCapabilityRegistry` | Builds compact capability context for the model.                                                                                                                                 | Phase 2 for local tools/skills; Phase 3 for MCP and provider-native capabilities | 8.8 `CapabilityRegistryEntry`; 8.14 `ContextAssembly`                                                                  | `services/capability_registry.py`; `services/context_manager.py`                               |
| `RenderToolsForProvider`   | Converts neutral tool definitions into provider-specific tool schemas.                                                                                                           | Phase 2                                                                          | 8.8 `ToolDefinition`; 8.11 `ModelRequest`; 8.12 `AnthropicRuntimeOptions` for Anthropic adapter options                | `llm/renderer.py`; `integrations/anthropic/tool_schemas.py`                                    |
| `ExecuteToolCall`          | Executes model-requested local tools and captures results for reasoning and memory.                                                                                              | Phase 2                                                                          | 8.11 `ToolCall`, `ToolResult`                                                                                          | `services/tool_runner.py`; `services/capability_loop.py`                                       |
| `ListMcpServers`           | Lists external MCP integrations.                                                                                                                                                 | Phase 3                                                                          | 8.8 `McpServerRecord`                                                                                                  | `services/mcp_manager.py`; `repositories/mcp.py`                                               |
| `ReadMcpResource`          | Loads memory-adjacent context from external MCP systems.                                                                                                                         | Phase 3                                                                          | 8.8 `McpResource`, `McpResourceTemplate`; 8.9 `RetrievedContextItem`                                                   | `services/mcp_manager.py`; `integrations/mcp/resources.py`                                     |
| `RenderMcpForProvider`     | Converts neutral MCP configuration into provider-native or fallback tool exposure.                                                                                               | Phase 3                                                                          | 8.8 `McpServerRecord`; 8.12 `AnthropicMcpServerConfig`, `AnthropicMcpToolsetConfig`                                    | `services/mcp_manager.py`; `integrations/anthropic/mcp_connector.py`                           |
| `AssembleContext`          | Selects, orders, and renders memory into the model-facing context.                                                                                                               | Phase 1                                                                          | 8.9 `RetrievedContextItem`; 8.14 `ContextAssembly`                                                                     | `services/context_manager.py`; `schemas/context.py`                                            |
| `RenderModelRequest`       | Converts assembled memory context into a provider request.                                                                                                                       | Phase 1                                                                          | 8.11 `ModelRequest`; 8.14 `ContextAssembly`; 8.12 `AnthropicRuntimeOptions` for Anthropic adapter options              | `llm/renderer.py`; `integrations/anthropic/messages_renderer.py`                               |
| `CallModel`                | Calls the selected model with memory-aware context.                                                                                                                              | Phase 1                                                                          | 8.11 `ProviderOptions`, `ModelRequest`, `ModelResponse`                                                                | `services/model_gateway.py`; `llm/base.py`; `integrations/anthropic/client.py`                 |
| `SubmitToolResults`        | Feeds tool results back into the model and later memory-write path.                                                                                                              | Phase 2                                                                          | 8.11 `ToolResult`, `ModelRequest`, `ModelResponse`; 8.13 `TurnMachineContext`; 8.16 `SubmitToolResultsRequest`         | `services/capability_loop.py`; `integrations/anthropic/tool_loop.py`                           |
| `ScoreMemoryImportance`    | Prevents low-value content from polluting durable memory.                                                                                                                        | Phase 1                                                                          | 8.10 `MemoryScoreComponents`; 8.14 `MemoryWriteCandidate`                                                              | `services/importance_scorer.py`                                                                |
| `ProposeMemoryWrites`      | Creates candidate profile, episode, summary, knowledge, or cache writes.                                                                                                         | Phase 1                                                                          | 8.14 `MemoryWriteCandidate`                                                                                            | `services/memory_writer.py`                                                                    |
| `CommitMemoryWrites`       | Persists approved memory writes and updates required indexes.                                                                                                                    | Phase 1                                                                          | 8.14 `CommittedMemoryWrite`                                                                                            | `services/memory_writer.py`; `repositories/memory_writes.py`                                   |
| `RankMemory`               | Orders candidate memories by relevance, importance, recency, metadata match, and actionability.                                                                                  | Phase 1, rule-based first                                                        | 8.10 `MemoryRankingPolicy`, `MemoryRankedItem`, `MemoryScoreComponents`                                                | `services/memory_ranker.py`                                                                    |
| `CheckMemoryConflicts`     | Detects stale, duplicate, or contradictory memories.                                                                                                                             | Phase 2                                                                          | 8.10 `MemoryConflict`, `ConflictCheckResult`                                                                           | `services/conflict_checker.py`                                                                 |
| `ConsolidateMemory`        | Merges overlapping memories while preserving source provenance.                                                                                                                  | Phase 2                                                                          | 8.10 `ConsolidationRun`; 8.1 `MemoryMetadata`                                                                          | `services/consolidation.py`; `workers/consolidate.py`                                          |
| `DispatchHook`             | Runs extension logic around memory retrieval and memory writes.                                                                                                                  | Phase 1, memory-specific hooks only                                              | 8.11 `HookEvent`, `HookDefinition`, `HookResult`                                                                       | `services/hook_manager.py`; `schemas/hooks.py`                                                 |
| `WriteAuditEvent`          | Records why memory was retrieved, injected, skipped, written, or rejected.                                                                                                       | Phase 1                                                                          | 8.14 `AuditEvent`                                                                                                      | `services/audit_logger.py`; `repositories/audit.py`                                            |
| `RunEvaluation`            | Tests retrieval quality, ranking, context assembly, and write policy.                                                                                                            | Phase 2                                                                          | 8.15 `EvaluationCase`, `EvaluationMetricResult`, `EvaluationRun`                                                       | `services/evaluation.py`; `repositories/evaluations.py`; `workers/evaluate.py`                 |

Phase 1 should prioritize the operations needed by a single-pass, memory-aware turn with a persisted state machine and basic knowledge/vector retrieval. It should not require extra LLM calls for interpretation, importance scoring, or summarization. Phase 2 should add post-Phase 1 retrieval quality and local capability support, but should still avoid MCP and provider-native features unless a specific use case depends on them. Do not require model-visible tools, native Skills, MCP, or conversation summarization for the first version unless a specific use case depends on them.

## 3. Agent Runtime Design

The agent runtime is the provider-neutral orchestration layer. It coordinates the turn state machine, model provider adapter, capability loop, tools, skills, MCP, hooks, memory retrieval, memory writing, and audit logging. Anthropic is the first implementation target, but these runtime concepts should not depend on Anthropic-specific wire formats.

### 3.1. State machine

Model each agent turn as a persisted state machine rather than a single blocking function call. This makes tool calls, user interrupts, retries, approval gates, memory writes, and recovery explicit.

Use one state vocabulary across the document: the `TurnState` enum in Section 8.13. The external lifecycle status in `TurnRecord.status` is only a UI/API projection over that internal state.

The state machine should store state transitions, inputs, outputs, errors, provider request ids, capability call ids, and audit events. This is useful for resumability, debugging, observability, deterministic tests, and safe interruption while tools are running.

Separate coarse turn status from internal executor state. `TurnRecord.status` is the external lifecycle status used by UI, APIs, and reports: `received`, `running`, `completed`, `failed`, or `cancelled`. `TurnMachineContext.state` is the persisted internal state from Section 8.13 that drives resumable execution: `START`, `INTERPRET_REQUEST`, `RETRIEVE_CONTEXT`, `ASSEMBLE_CONTEXT`, `RENDER_MODEL_REQUEST`, `CALL_MODEL`, `HANDLE_MODEL_RESPONSE`, `EXECUTE_TOOL_CALLS`, `APPEND_TOOL_RESULTS`, `WRITE_MEMORY`, `FINALIZE`, `FAILED`, and `CANCELLED`.

Recommended provider-neutral transition model:

```text
START -> INTERPRET_REQUEST

INTERPRET_REQUEST -> RETRIEVE_CONTEXT
RETRIEVE_CONTEXT -> ASSEMBLE_CONTEXT
ASSEMBLE_CONTEXT -> RENDER_MODEL_REQUEST
RENDER_MODEL_REQUEST -> CALL_MODEL
CALL_MODEL -> HANDLE_MODEL_RESPONSE

HANDLE_MODEL_RESPONSE -> EXECUTE_TOOL_CALLS
  when the normalized model response requests capabilities and iteration < max_iterations

HANDLE_MODEL_RESPONSE -> WRITE_MEMORY
  when the normalized model response is final and memory-write policy is enabled

HANDLE_MODEL_RESPONSE -> FINALIZE
  when the normalized model response is final and memory-write policy is disabled

HANDLE_MODEL_RESPONSE -> FAILED
  when required capabilities cannot run, policy blocks continuation, or max_iterations is reached

EXECUTE_TOOL_CALLS -> APPEND_TOOL_RESULTS
APPEND_TOOL_RESULTS -> RENDER_MODEL_REQUEST

WRITE_MEMORY -> FINALIZE
  even when no memory-write candidates are produced

any non-terminal state -> FAILED
  on unrecoverable exception

any non-terminal state -> CANCELLED
  on user or system interrupt
```

Terminal state mapping: `FINALIZE` is the successful terminal executor state and should set `TurnRecord.status` to `completed`; `FAILED` should set it to `failed`; `CANCELLED` should set it to `cancelled`. `ResumeTurn`, `SubmitToolResults`, and `SubmitApproval` should only operate on non-terminal states.

Iteration semantics: `iteration` counts model/capability round trips, not every state transition. Increment it when the state machine re-enters `HANDLE_MODEL_RESPONSE` after a model call. A single Phase 1 turn with no tool calls normally uses one iteration.

Implementation rules:

- Persist `TurnMachineContext` after every transition so a failed worker can resume from the last completed state.
- Record every transition as a `StateTransitionRecord` and `AuditEvent` with `turn_id`, previous state, next state, reason, and relevant metadata.
- Keep provider-specific rendering and response parsing behind the selected model adapter; the state machine should operate on `ModelRequest`, `ModelResponse`, `ToolCall`, and `ToolResult`.
- Bound model/capability loops with `max_iterations`; hitting the limit should produce `FAILED`, not an unbounded loop.
- Classify errors as transient or terminal. Transient provider/network errors should retry with bounded exponential backoff before transition to `FAILED`.
- Treat hook vetoes as transition guards. For example, a `before_tool_call` hook can block a sensitive tool and move the turn to `FAILED` or return a controlled capability result for the next model call.
- Persist raw provider request and response payloads for debugging and audit, but expose normalized contracts to the rest of the runtime.
- Avoid treating per-transition writes as free in production. SQLite/local-first can persist after every transition; remote Postgres or shared databases may batch transition records or asynchronously flush non-critical audit details while keeping the current `TurnMachineContext` durable.
- Define a concurrent-turn policy per session: either serialize turns for a session or require optimistic concurrency tokens on memory writes and summary updates.

Implementation note for Phase 1 scope: include a persisted turn state machine in Phase 1, but start with the core memory path: `START`, `INTERPRET_REQUEST`, `RETRIEVE_CONTEXT`, `ASSEMBLE_CONTEXT`, `RENDER_MODEL_REQUEST`, `CALL_MODEL`, `HANDLE_MODEL_RESPONSE`, `WRITE_MEMORY`, `FINALIZE`, `FAILED`, and `CANCELLED`. Keep `EXECUTE_TOOL_CALLS` and `APPEND_TOOL_RESULTS` as Phase 2 states for the capability loop. This keeps the first implementation auditable and resumable without requiring tools, MCP, or executable skills.

### 3.2. Capability loop

The capability loop is the runtime loop that lets the model request external actions and then continue reasoning with the results. It should be provider-neutral even when the first implementation uses Anthropic Messages API `tool_use` and `tool_result` blocks.

Loop outline:

1. Assemble context and render a provider request.
2. Call the model provider.
3. Normalize the response into assistant content plus requested capabilities.
4. If no capability is requested, finish the turn and run memory-write policy.
5. If a capability is requested, validate permissions, dispatch the capability, append the result to the next model input, and continue.
6. Stop on final response, user interrupt, error policy, approval denial, or iteration limit.

The loop should treat tools, skills/procedures, MCP, and provider-native features as capability sources behind one dispatch interface. Provider adapters decide how each capability is rendered for a specific LLM API.

Implementation note for Phase 1 scope: the capability loop is not required for the first agentic-memory loop. A minimal implementation can retrieve memory, assemble context, call the model once, and write selected memory after the turn. Add the capability loop in Phase 2 when the model needs to request local visible tools, executable local skills, approval-gated actions, or mid-turn external operations whose results should feed back into reasoning and memory. Add MCP and provider-native capability sources in Phase 3.

### 3.3. Tools

Tools are callable actions exposed to the model with schemas, descriptions, permissions, and execution handlers. The runtime should keep a provider-neutral representation such as `ToolDefinition`, `ToolCall`, and `ToolResult`, then let provider adapters convert those records into Anthropic, OpenAI-compatible, or local test formats.

Tool execution should be local by default for this system: Python functions, shell-safe wrappers, file operations, database accessors, or service clients. Provider-native or server-side tools should be normalized by provider adapters into the same `ToolCall` and `ToolResult` concepts, even when execution does not happen inside the local `ToolRunner`. Tool results should include structured output, display text, errors, side-effect metadata, and audit references.

Implementation note for Phase 1 scope: model-visible tools are not required for the first agentic-memory loop. The minimal loop can retrieve memory, assemble context, call the model, and write memory through internal runtime services without exposing callable tools to the model. Add model-visible local tools in Phase 2 when the agent needs to act on external state, read files, query databases, call APIs, execute code, or request controlled memory operations directly. Provider-native or server-side tools belong in Phase 3 after local tool semantics are stable.

### 3.4. Skills

Skills are procedural memory packaged as reusable instructions, workflows, examples, scripts, or assets. The runtime should expose compact skill metadata during context assembly and load full skill content only when selected by routing policy or by the model.

The same local skill registry should support multiple providers:

- With Anthropic Skills support, selected local skills can be mapped into Anthropic skill configuration.
- Without native skill support, selected skill instructions can be injected into provider-neutral context.
- If a skill includes executable helpers, those helpers should be exposed through the tool/capability dispatcher rather than hidden inside prompt text.

Skills are not just prompt snippets; they are part of procedural memory and should have ids, versions, scopes, activation rules, dependencies, and audit traces.

Implementation note for Phase 1 scope: provider-native Skills are not required for the minimal agentic-memory loop, but local procedural memory is required if the system should remember reusable ways of working. Start with local `Procedure` records for playbooks, policies, workflows, and checklists; retrieve relevant procedures during context assembly; and inject selected procedure text into the model context. Add a local skill registry and executable skill assets in Phase 2 when procedure packaging and execution become necessary. Add Anthropic Skills mapping in Phase 3 as a provider-native integration.

### 3.5. MCP

MCP exposes external tools, resources, prompts, and resource templates through a standard protocol. The runtime should model MCP servers as capability providers, not as a separate memory type.

There are two useful integration paths:

- Native provider connector: render remote MCP server configuration for providers that can connect directly.
- Local MCP client: connect from the Python runtime and wrap MCP tools/resources as provider-neutral capabilities.

For agentic memory, MCP is most valuable when memory-adjacent resources live outside the core database: project files, documents, knowledge bases, ticket systems, calendars, or specialized domain systems. MCP calls should produce normal capability results and audit events so they can participate in context assembly, episodic memory, and later evaluation.

Implementation note for Phase 1 scope: MCP is not required for the minimal agentic-memory loop. Start with local/internal memory interfaces for profile memory, recent messages, simple episodes, local procedures, knowledge items, basic vector or hybrid search, context assembly, model calls, selective memory writes, the persisted turn state machine, audit events, and memory-specific hooks. Add advanced retrieval quality, summaries, reranking, lifecycle management, and richer local capability support in Phase 2. Add MCP in Phase 3 only when memory-adjacent resources live outside the core system; this includes MCP registry, local MCP client wrappers, fallback MCP tools, and provider-native MCP connectors.

### 3.6. Hooks

Hooks are runtime extension points, not a separate memory type. They observe or control the state machine, capability loop, memory lifecycle, provider calls, and error handling.

Group hook events by their primary relationship to agentic memory. The grouping is descriptive; all hook events should still use one typed hook bus and write audit events.

Memory-specific hooks operate directly on memory retrieval or memory persistence:

| Event | Relationship to agentic memory |
|---|---|
| `before_memory_retrieval` | Adjusts memory scope, filters, privacy constraints, retrieval strategy, and recall budget before profile, episode, procedure, knowledge, vector, or graph memory is queried. |
| `after_memory_retrieval` | Reviews retrieved memories before injection; can rerank, drop stale or unsafe records, attach provenance, and record retrieval quality for evaluation. |
| `before_memory_write` | Validates candidate writes before persistence; can deduplicate, detect conflicts, require approval, reject low-value memories, and enforce retention policy. |
| `after_memory_write` | Updates secondary indexes, emits audit records, schedules consolidation, and records what durable memory changed because of the turn. |

Memory-adjacent hooks do not read or write memory by themselves, but they strongly influence what memory is selected, loaded, or transformed:

| Event | Relationship to agentic memory |
|---|---|
| `before_request_interpretation` | Shapes the interpretation step so the runtime extracts entities, task type, user intent, constraints, and likely memory needs from the incoming request. |
| `after_request_interpretation` | Converts interpretation output into a retrieval plan, capability plan, and possible memory-write expectations for the rest of the turn. |
| `before_context_assembly` | Controls which retrieved memories, summaries, procedures, MCP resources, recent turns, and instructions are eligible for the context window. |
| `after_context_assembly` | Verifies that injected memory is relevant, ordered, source-labeled, within budget, and not contradictory or overexposed. |
| `before_skill_load` | Applies procedural-memory routing before loading a skill; can enforce version, scope, dependency, and permission rules. |
| `after_skill_load` | Records which procedure or skill became active context and can expose related tools, examples, or audit metadata. |
| `before_mcp_call` | Applies scope, authorization, provenance, and resource-selection policy before accessing external memory-adjacent systems. |
| `after_mcp_call` | Converts MCP outputs into context items, knowledge candidates, episode evidence, or audit records when the external result is useful later. |

General agent-runtime hooks control the broader turn loop. Their memory relationship is indirect, but they still affect memory quality, provenance, and recovery:

| Event | Relationship to agentic memory |
|---|---|
| `before_model_call` | Performs a final check of provider-bound context, including whether injected memories are necessary, compact, source-labeled, and safe to send. |
| `after_model_call` | Captures model output, tool requests, usage, and stop reason so the runtime can propose summaries, episodes, observations, or audit records. |
| `before_tool_call` | Records the planned action, validates permissions and side effects, and preserves enough context to reconstruct the episode later. |
| `after_tool_call` | Captures tool results, errors, costs, and side effects that may become episode records, knowledge items, or memory-write candidates. |
| `after_turn` | Finalizes the turn by triggering summarization, episode capture, selective memory writing, cleanup, and evaluation events. |
| `on_error` | Preserves failure context, prevents partial memory corruption, records failed episodes or recovery lessons, and supports resumable turns. |

Hooks should receive typed event payloads, return optional mutations or vetoes where allowed, and always write audit events. Keep them deterministic where possible; if a hook calls an LLM or external service, record that as a capability call.

Side note for Phase 1 scope: implement the memory-specific hooks first. `before_memory_retrieval`, `after_memory_retrieval`, `before_memory_write`, and `after_memory_write` are the only hook events required to start a working agentic-memory loop because they guard the two critical memory boundaries: reading memory into context and writing new memory back to storage. Add memory-adjacent and general agent-runtime hooks in Phase 2 for better routing, provenance, observability, and recovery. Add MCP-specific hooks in Phase 3 with MCP integration.

## 4. Deployment and Protocol Baseline

Use this as the deployment and protocol posture for the Python implementation in Section 5. The detailed protocol comparison remains in Section 11.

- Use a local agent runtime as the default assumption.
- Use HTTPS from the local runtime to a remote LLM provider when the model is remote.
- Keep tools, skills, MCP servers, workspace access, and the turn state machine local unless a clear product requirement justifies moving them to the cloud.
- Use local function calls for services and tools where possible.
- Use SQLite, local Postgres, or embedded/vector storage for private local memory.
- Use a remote memory service only when cross-device continuity, team sharing, centralized backup, or audit governance matters.
- Use a local job runner for background memory lifecycle work by default.
- Add SSE, WebSocket, or a local event bus only for live turn observation or user intervention.
- Add a queue/event bus when lifecycle jobs need durability across crashes, retries, or multiple workers.
- Use MCP or stdio/subprocess for local tool and resource integration.
- Add gRPC only when REST, local HTTP, or in-process calls become a measurable bottleneck.
- Keep the logical API contracts independent from the transport.

## 5. Python implementation target

### 5.1. Recommended baseline

The baseline should support a local-first, provider-neutral runtime while still making Anthropic the first concrete model adapter. Not every item has to be enabled immediately; each item is chosen because it maps cleanly to the domain model, logical API, and package layout.

- Python 3.12+ because the implementation benefits from modern typing, dataclasses/Pydantic ergonomics, async support, and a stable dependency ecosystem.
- Pydantic v2 because the system needs explicit data contracts for domain objects, logical operation payloads, provider adapter inputs, validation, serialization, and test fixtures.
- `anthropic` Python SDK because Anthropic Messages is the first target runtime, and using the official SDK reduces request/response drift for content blocks, usage, errors, and future Anthropic-specific options.
- SQLAlchemy 2.x plus Alembic because source-of-truth memory records, turns, audit events, summaries, and procedures need durable relational storage with explicit migrations.
- SQLite plus a local vector adapter for the default local-first implementation because it keeps private memory, turn state, and prototype setup simple.
- Local PostgreSQL with pgvector for production-like local deployments because it can store structured memory records and vector indexes in one operational database.
- Remote PostgreSQL, vector databases, or a dedicated memory service only when cross-device continuity, team sharing, centralized backup, audit governance, or multi-runtime coordination justifies moving memory out of the local runtime.
- ChromaDB as a local development vector adapter because it is convenient for prototyping semantic retrieval, but it should not be the only source of truth for memory records.
- Plain files or object storage because append-only logs, exports, snapshots, human-readable memory records, and large artifacts are easier to inspect and archive outside relational tables.
- FastAPI because it is a practical HTTP transport adapter when the runtime needs to be exposed to a web UI or another service; keep it optional so the core runtime can also run in-process.
- Redis because session cache, retrieval cache, queue coordination, and distributed locks may become useful under load, but adding Redis too early increases operational complexity.
- Background workers because summarization, embedding generation, consolidation, index repair, exports, and evaluations may become long-running or retryable jobs that should not block the turn loop.
- MCP clients because MCP is useful for external tools and resources, but it should be added only when memory-adjacent data lives outside the core system; local function calls or stdio/subprocess are simpler when tools live in the same runtime.
- Anthropic Skills, provider-native/server-side tools, and Claude Code compatibility because they can improve provider integration and agent ergonomics, but the local procedure/tool registry should remain the source of truth and these features should be adapter renderings.

Design principle: keep memory, retrieval, context assembly, hooks, and capability orchestration provider-neutral. Keep Anthropic-specific payloads, beta headers, content blocks, local tool schemas, tool-loop rendering, Skills containers, server-side tools, and MCP connector config inside Anthropic integration modules rather than the domain model, logical API, or core services.

### 5.2. Runtime layers

Keep the runtime split into these layers:

- `schemas`: Pydantic data contracts for neutral domain objects and logical operation payloads. These contracts should stay provider-neutral and transport-neutral except for explicitly named adapter contracts.
- `services`: application logic for the logical API and runtime orchestration, including turn execution, state transitions, interpretation, retrieval, context assembly, model dispatch, capability dispatch, hooks, memory writes, lifecycle policy, and audit logging.
- `repositories`: persistence adapters for source-of-truth records, messages, memories, episodes, procedures, summaries, audit events, vector indexes, key-value caches, files, evaluations, and integration registries.
- `llm`: provider-neutral model interfaces, request renderers, provider clients, response normalizers, usage normalization, and test/noop providers.
- `embeddings`: embedding provider interfaces and concrete embedding clients used by vector indexing and semantic retrieval. Keep embedding generation separate from memory persistence so index backends and embedding models can change independently.
- `providers`: registries and selection policy for LLM providers, embedding providers, vector backends, and runtime adapters. This layer chooses implementations; it should not contain provider-specific wire-format logic.
- `integrations`: provider-specific and external-system adapters, including Anthropic Messages rendering, Anthropic tool/skill/MCP mappings, local MCP clients, Claude Code compatibility, files, and other external capability connectors.
- `api`: optional transport adapters such as FastAPI route handlers. Keep this layer thin; it should translate transport concerns into service calls and should not contain memory policy.
- `workers`: optional asynchronous jobs for summarization, embedding generation, consolidation, index repair, evaluation, exports, and other long-running maintenance tasks.
- `db`: SQLAlchemy models, Alembic migrations, database sessions, and storage bootstrap code. Keep database-specific schemas here, not in the neutral domain or logical API contracts.
- `tests`: unit, integration, fixture, and evaluation tests for retrieval, ranking, context assembly, state transitions, provider adapters, memory writes, hooks, and persistence behavior.

### 5.3. Suggested package layout by phase

Implement the package layout cumulatively. Phase 1 should be runnable on its own; Phase 2 and Phase 3 add modules without changing the core contracts.

Phase 1: core memory loop with persisted state machine, profile memory, recent messages, simple episodes, local procedures, knowledge/vector retrieval, context assembly, one Anthropic Messages model call, heuristic interpretation/scoring, selective memory writes, memory-specific hooks, and audit events.

```text
agentic_memory/
  schemas/
    base.py
    actors.py
    messages.py
    turns.py
    interpretation.py
    retrieval.py
    memory_writes.py
    memory.py
    model.py
    knowledge.py
    episodes.py
    procedures.py
    context.py
    providers.py
    hooks.py
    audit.py
  llm/
    base.py
    renderer.py
    anthropic_messages.py
    noop.py
  services/
    turn_runner.py
    turn_state_machine.py
    interpreter.py
    retrieval.py
    context_manager.py
    model_gateway.py
    memory_writer.py
    memory_ranker.py
    importance_scorer.py
    hook_manager.py
    audit_logger.py
  integrations/
    anthropic/
      contracts.py
      client.py
      messages_renderer.py
      content_blocks.py
  repositories/
    turns.py
    sessions.py
    messages.py
    profile_memory.py
    knowledge.py
    episodes.py
    procedures.py
    memory_writes.py
    audit.py
    vector_index.py
    chroma_vector.py
    files.py
  embeddings/
    base.py
    local.py
  providers/
    registry.py
  db/
    models.py
    migrations/
  tests/
```

Phase 2: memory quality, lifecycle management, summaries, expanded hook event coverage, local capability abstraction, local model-visible tools, local skill packaging, Anthropic local tool-call rendering, evaluation, and background jobs.

```text
agentic_memory/
  schemas/
    summaries.py
    capabilities.py
    tools.py
    skills.py
    evaluations.py
  llm/
    openai_compatible.py
  services/
    summarizer.py
    memory_lifecycle.py
    conflict_checker.py
    consolidation.py
    capability_loop.py
    capability_registry.py
    tool_runner.py
    skill_manager.py
    evaluation.py
  integrations/
    anthropic/
      tool_schemas.py
      tool_loop.py
  repositories/
    summaries.py
    evaluations.py
    kv.py
    pgvector_index.py
  embeddings/
    openai.py
  workers/
    summarize.py
    embed.py
    consolidate.py
    evaluate.py
```

Phase 3: MCP, provider-native capabilities, Anthropic Skills, provider-native/server-side tools, Claude Code compatibility, and optional external transport adapters.

```text
agentic_memory/
  api/
    routes_turns.py
    routes_context.py
    routes_memory.py
    routes_capabilities.py
    routes_hooks.py
    routes_audit.py
  schemas/
    mcp.py
  services/
    mcp_manager.py
  integrations/
    anthropic/
      server_tools.py
      skills.py
      mcp_connector.py
      files.py
    mcp/
      client.py
      resources.py
      tools.py
    claude_code/
      hooks.py
      skill_loader.py
  repositories/
    mcp.py
```

Alignment note: the package layout mirrors Sections 1-3. `schemas` implements domain contracts, `services` implements logical operations and the runtime state machine, `repositories` persists memory and audit records, and `integrations` contains provider or protocol adapters. Phase 1 must not depend on `api`, model-visible tools, native Skills, MCP, distributed workers, or provider-native capability connectors.


## 6. Provider-Neutral LLM Architecture

The runtime should treat LLMs as replaceable adapters. Core services should speak in neutral contracts:

- `ModelRequest`: system instructions, messages, tools, capability registry, retrieved context, provider options
- `ModelResponse`: assistant content, tool requests, usage, stop reason, raw provider payload
- `ToolCall`: provider-neutral callable action request
- `ToolResult`: provider-neutral action result
- `ModelProvider`: provider-neutral adapter protocol that renders requests, calls the selected model, and normalizes responses
- `ProviderRegistry`: registry that selects the concrete provider adapter for a provider name, model, and runtime configuration

`services/model_gateway.py` should orchestrate provider selection and model calls. `llm/base.py` should define provider protocols, `llm/renderer.py` should hold shared provider-neutral rendering helpers, `providers/registry.py` should select adapters, and `integrations/` should contain provider-specific wire-format conversion.

Anthropic should be the first and most complete adapter. Other providers can be partial adapters as long as they implement the same `ModelProvider` protocol and normalize their provider-specific payloads back into the same neutral contracts.

Provider-specific behavior belongs in `llm/` and `integrations/` adapter modules:

- Anthropic content block conversion
- Anthropic `tools` schema conversion
- Anthropic `tool_use` parsing and `tool_result` rendering
- Anthropic server-tool or beta header configuration
- Anthropic Skills container configuration
- Anthropic MCP connector configuration
- Provider-specific token accounting and usage normalization

Provider-neutral behavior belongs in `services/`:

- memory retrieval
- ranking and compression
- capability selection
- hook dispatch
- memory write policy
- audit logging
- turn state-machine progression, terminal-state mapping, and iteration bounds

## 7. Anthropic Adapter Mapping

Use Anthropic Messages API as the first concrete provider adapter. This section maps the provider-neutral contracts and runtime concepts from Sections 1-6 into Anthropic-specific request, response, tool, skill, and MCP shapes.

Do not let this adapter mapping change the implementation stages. Phase 1 only needs a single Anthropic Messages call. Phase 2 adds local tool rendering and the `tool_use` / `tool_result` loop. Phase 3 adds Anthropic-native Skills, MCP connector config, server-side tools, and other provider-native capabilities.

The Anthropic adapter should support these mappings by stage:

- **Messages request/response**: render `ModelRequest` into Anthropic `system`, `messages`, model options, and content blocks; normalize Anthropic output back into `ModelResponse`.
- **Client tools**: map neutral `ToolDefinition` records to Anthropic `tools`; map Anthropic `tool_use` blocks to `ToolCall`; map `ToolResult` records back to Anthropic `tool_result` blocks.
- **Skills**: map selected local `ProcedureRecord` or `CapabilityRegistryEntry` records into Anthropic Skills configuration only when provider-native skill packaging is useful. The local procedure/skill registry remains the source of truth.
- **MCP connectors**: map `McpServerRecord` and MCP capability records into Anthropic MCP connector settings only when external resources justify MCP. Local MCP clients can still be wrapped as neutral tools.
- **Hooks**: keep hook dispatch in the provider-neutral `services/hook_manager.py`; the Anthropic adapter should only provide provider payloads and normalized events for hook payloads.

The Anthropic adapter should preserve the provider-neutral state machine from Section 3. In Anthropic-specific terms:

1. Assemble provider-neutral context.
2. Render an Anthropic Messages request with `system`, `messages`, `tools`, and Anthropic-specific options.
3. Send the request through the Anthropic client adapter.
4. Normalize the Anthropic response into `ModelResponse`.
5. For Phase 1, return the final response to the neutral memory-write path.
6. For Phase 2 tool loops, if the response contains `tool_use`, return neutral `ToolCall` records to the persisted state machine.
7. Let the state machine dispatch those calls through `ToolRunner`, then ask the Anthropic adapter to render matching `tool_result` content blocks for the next model request.
8. Let neutral services dispatch post-turn hooks and memory-write policy.

Anthropic-specific loop responsibilities should stay inside the adapter:

- Render neutral `ModelRequest` fields into Anthropic `system`, `messages`, `tools`, model options, optional container settings, optional native Skills configuration, optional `mcp_servers`, and optional MCP toolsets.
- Parse Anthropic `tool_use` blocks into neutral `ToolCall` records; do not execute tools or advance the state machine directly from the adapter.
- Render neutral `ToolResult` records back into Anthropic `tool_result` content blocks when the state machine resumes the model.
- Normalize final Anthropic content, stop reasons, usage, errors, and provider request ids into `ModelResponse`.
- Save rendered Anthropic payloads and raw responses for audit/debugging, but keep retrieval, ranking, hook dispatch, tool execution, MCP fallback, and memory writes provider-neutral.

Keep Skills and MCP as first-class capabilities in the neutral runtime, but treat Anthropic-native Skills and MCP connector payloads as adapter renderings. If a provider does not support those features natively, the runtime can still load a local skill into prompt context or call a local MCP client through a wrapped tool.

## 8. Python Data Contracts

These Pydantic schemas are the Python implementation mapping of the neutral domain model in Section 1 and the logical API in Section 2. They are not a separate language-neutral core data model.

Use these contracts as the boundary between API routes, services, repositories, and provider adapters. Keep them provider-neutral where possible. Provider-specific payloads, Anthropic content blocks, OpenAI tool-call shapes, HTTP details, SQLAlchemy persistence models, and database-specific columns should stay in adapters or persistence mappings. SQLAlchemy models can be separate persistence models, but they should map cleanly to these contracts.

### 8.1. Shared schema base

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


Scope = Literal["user", "project", "session", "tenant", "global"]
MemorySourceType = Literal["message", "tool_result", "document", "episode", "manual", "system"]
Priority = Literal["low", "medium", "high"]
PrivacyClassification = Literal["public", "internal", "sensitive", "pii", "financial", "health", "legal"]
RetentionAction = Literal["retain", "expire", "archive", "delete", "review"]


class ActorRef(Schema):
    tenant_id: str
    user_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None


class SourceRef(Schema):
    type: MemorySourceType
    id: str | None = None
    uri: str | None = None
    timestamp: datetime


class ProvenanceMetadata(Schema):
    scope: Scope
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    source: SourceRef
    created_at: datetime
    updated_at: datetime


class MemoryLifecycle(Schema):
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    expires_at: datetime | None = None
    retention_action: RetentionAction = "retain"


class MemoryRelations(Schema):
    supersedes_id: str | None = None
    superseded_by_id: str | None = None
    related_ids: list[str] = Field(default_factory=list)


class ConsentRecord(Schema):
    actor: ActorRef
    allowed_classifications: list[PrivacyClassification] = Field(default_factory=list)
    purpose: str
    granted_at: datetime
    revoked_at: datetime | None = None


class RetentionPolicy(Schema):
    scope: Scope
    default_action: RetentionAction = "retain"
    expires_after_days: int | None = Field(default=None, ge=1)


class PrivacyPolicy(Schema):
    classification: PrivacyClassification = "internal"
    consent_id: str | None = None
    retention_policy_id: str | None = None
    can_export: bool = True
    can_forget: bool = True


class MemoryMetadata(Schema):
    provenance: ProvenanceMetadata
    lifecycle: MemoryLifecycle = Field(default_factory=MemoryLifecycle)
    relations: MemoryRelations = Field(default_factory=MemoryRelations)
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
```

`MemoryMetadata` is an aggregate for convenience, not a single database row that must be updated on every read. Store immutable provenance with the memory record, and store lifecycle/access counters separately when the backend would otherwise create write contention. Repository queries must always enforce `ActorRef` scoping and privacy policy before returning memory.

### 8.2. Conversation message

```python
MessageRole = Literal["user", "assistant", "system", "tool"]
ContentBlockKind = Literal["text", "tool_use", "tool_result", "image", "document"]


class ContentBlock(Schema):
    kind: ContentBlockKind
    text: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(Schema):
    message_id: str
    session_id: str
    role: MessageRole
    content: list[ContentBlock]
    tool_call_id: str | None = None
    tool_name: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    created_at: datetime
```

### 8.3. Structured conversation summary

```python
EpisodeOutcome = Literal["success", "failure", "partial", "unknown"]


class EpisodeNote(Schema):
    event: str
    outcome: EpisodeOutcome
    lesson: str | None = None


class ConversationSummary(Schema):
    summary_id: str
    session_id: str
    covered_message_ids: list[str]
    goal: str | None = None
    decisions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    durable_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_step: str | None = None
    episode_notes: list[EpisodeNote] = Field(default_factory=list)
    token_count: int = Field(ge=0)
    created_at: datetime
```

### 8.4. Profile memory

```python
class ProfileMemory(Schema):
    memory_id: str
    actor: ActorRef
    key: str
    value: Any
    description: str | None = None
    metadata: MemoryMetadata
```

### 8.5. External knowledge item

```python
ContentType = Literal["text", "markdown", "json", "pdf", "html", "csv", "other"]


class KnowledgeDocument(Schema):
    document_id: str
    actor: ActorRef
    title: str | None = None
    source_uri: str | None = None
    content_type: ContentType
    text: str | None = None
    metadata: MemoryMetadata


class KnowledgeChunk(Schema):
    chunk_id: str
    document_id: str
    text: str
    ordinal: int = Field(ge=0)
    token_count: int | None = Field(default=None, ge=0)
    embedding_id: str | None = None
    metadata: MemoryMetadata
```

### 8.6. Vector index adapter contracts

```python
VectorItemKind = Literal["profile", "summary", "knowledge", "episode", "procedure"]


class VectorUpsertRequest(Schema):
    item_id: str
    kind: VectorItemKind
    text: str
    embedding: list[float] | None = None
    embedding_model: str
    vector_dim: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorUpsertResult(Schema):
    item_id: str
    embedding_id: str
    embedding_model: str
    vector_dim: int = Field(ge=1)


class VectorSearchRequest(Schema):
    query: str
    kinds: list[VectorItemKind] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    strategy: Literal["vector", "keyword", "hybrid"] = "vector"
    limit: int = Field(default=10, ge=1, le=100)


class HybridSearchRequest(Schema):
    query: str
    kinds: list[VectorItemKind] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    embedding_model: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class VectorSearchResult(Schema):
    item_id: str
    kind: VectorItemKind
    relevance: float = Field(ge=0.0, le=1.0)
    raw_score: float | None = None
    raw_score_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Every vector backend adapter should return normalized `relevance` in `0.0..1.0`. Preserve the backend-specific distance or similarity value in `raw_score` for debugging.

Protocol note: these `Request` and `Result` schemas are service/adapter contracts, not HTTP request/response contracts. The same shapes should work for in-process calls, REST, gRPC, queue jobs, or local worker calls. Protocol-specific concerns such as headers, auth tokens, pagination cursors, retry ids, HTTP status codes, streaming chunks, queue acknowledgements, and deadlines should live in transport envelopes or adapters, not in these vector contracts.

### 8.7. Episode record

```python
StoredEpisodeOutcome = Literal["success", "failure", "partial"]


class ActionRef(Schema):
    kind: Literal["tool_call", "model_call", "manual_step", "system_step"]
    name: str
    call_id: str | None = None
    summary: str | None = None


class TokenCost(Schema):
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)


class EpisodeRecord(Schema):
    episode_id: str
    actor: ActorRef
    task_type: str
    timestamp: datetime
    goal: str
    approach: str | None = None
    actions: list[ActionRef] = Field(default_factory=list)
    outcome: StoredEpisodeOutcome
    error_reason: str | None = None
    resolution: str | None = None
    lesson_learned: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    token_cost: TokenCost | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    embedding_id: str | None = None
    metadata: MemoryMetadata
```

### 8.8. Procedure and capability records

```python
ProcedureKind = Literal["skill", "playbook", "policy", "checklist", "workflow", "mcp-guide"]
CapabilityKind = Literal["tool", "skill", "procedure", "mcp_server", "mcp_resource", "mcp_template"]
McpConnectionKind = Literal["stdio", "http", "sse", "websocket", "provider_native"]


class ProcedureRecord(Schema):
    procedure_id: str
    name: str
    kind: ProcedureKind
    description: str
    trigger_conditions: list[str] = Field(default_factory=list)
    location: str | None = None
    content: str | None = None
    constraints: list[str] = Field(default_factory=list)
    metadata: MemoryMetadata


class ToolDefinition(Schema):
    tool_id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    provider_hint: str | None = None


class ProcedureTarget(Schema):
    kind: Literal["procedure"]
    procedure_id: str


class ToolTarget(Schema):
    kind: Literal["tool"]
    tool_id: str


class McpTarget(Schema):
    kind: Literal["mcp_server", "mcp_resource", "mcp_template"]
    server_id: str
    uri: str | None = None
    uri_template: str | None = None


class McpResourceTemplate(Schema):
    server_id: str
    name: str
    uri_template: str
    description: str
    input_schema: dict[str, Any] | None = None


class McpServerRecord(Schema):
    server_id: str
    name: str
    description: str | None = None
    connection_kind: McpConnectionKind
    connection_ref: str | None = None
    authorization_ref: str | None = None
    enabled: bool = True
    metadata: MemoryMetadata


class McpResource(Schema):
    resource_id: str
    server_id: str
    uri: str
    name: str | None = None
    content_type: ContentType
    text: str | None = None
    metadata: MemoryMetadata


class CapabilityRegistryEntry(Schema):
    capability_id: str
    kind: CapabilityKind
    name: str
    description: str
    trigger_conditions: list[str] = Field(default_factory=list)
    target: ProcedureTarget | ToolTarget | McpTarget | None = None
    schema: dict[str, Any] | None = None
```

Protocol note: `location`, `uri_template`, and capability `target` values are integration references, not transport contracts. Keep protocol-specific connection details, credentials, headers, process commands, and retry behavior in capability adapters or MCP/tool configuration, not in the reusable procedure and capability records.

Provider note: `provider_hint` is a non-binding adapter hint, not a core domain invariant. Keep `ProcedureRecord`, `ToolDefinition`, and `CapabilityRegistryEntry` neutral enough to work with Anthropic, OpenAI-compatible providers, local models, and test providers. Provider-specific schema restrictions, native tool formats, Anthropic Skills packaging, OpenAI tool metadata, and MCP toolset details should be produced by provider adapters from these neutral records.

### 8.9. Retrieved context item

```python
RetrievedContextKind = Literal[
    "profile",
    "summary",
    "knowledge",
    "episode",
    "procedure",
    "mcp_resource",
    "recent_turn",
]


class RetrievedContextItem(Schema):
    item_id: str
    kind: RetrievedContextKind
    title: str | None = None
    text: str
    source: SourceRef
    score: float | None = None
    score_components: MemoryScoreComponents | None = None
    priority: Priority
    token_count: int | None = Field(default=None, ge=0)
```

### 8.10. Memory scoring and lifecycle

```python
class MemoryScoreComponents(Schema):
    relevance: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    metadata_match: float = Field(default=0.0, ge=0.0, le=1.0)
    actionability: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryRankingPolicy(Schema):
    relevance_weight: float = 0.4
    importance_weight: float = 0.25
    recency_weight: float = 0.2
    metadata_match_weight: float = 0.1
    actionability_weight: float = 0.05
    decay_factor_per_hour: float = 0.995
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryRankedItem(Schema):
    item: RetrievedContextItem
    final_score: float = Field(ge=0.0, le=1.0)
    components: MemoryScoreComponents


MemoryConflictKind = Literal["duplicate", "contradiction", "stale", "superseded"]


class MemoryConflict(Schema):
    conflict_id: str
    kind: MemoryConflictKind
    candidate_id: str
    existing_memory_id: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_action: Literal["keep_both", "merge", "supersede_existing", "reject_candidate"]


class ConflictCheckResult(Schema):
    candidate_id: str
    conflicts: list[MemoryConflict] = Field(default_factory=list)
    allow_write: bool = True


class ConsolidationRun(Schema):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    actor: ActorRef | None = None
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    merged_count: int = Field(default=0, ge=0)
    created_count: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
```

Ranking rule: `final_score = sum(weight_i * component_i)` and implementations must validate that the active weights sum to `1.0` within a small tolerance. Phase 1 should support a basic policy such as relevance plus importance. Recency decay, actionability, learned reranking, and cross-encoder rerankers are Phase 2 quality improvements. The default `decay_factor_per_hour` is only a starting point and should be tuned per deployment.

Use this lifecycle model instead of storing every interaction blindly. The article's ChromaDB example is useful for demonstrating vector search, but production code should keep structured records in SQL and treat the vector index as a secondary index that can be rebuilt.

### 8.11. Provider-neutral model, tool, and hook contracts

```python
ProviderName = str
KnownProviderName = Literal["anthropic", "openai_compatible", "noop"]
StopReason = Literal[
    "finished",
    "length_limit",
    "requested_tool",
    "stop_sequence_matched",
    "refused",
    "error",
]


class ProviderOptions(Schema):
    name: ProviderName
    model: str
    context_window_hint: int | None = Field(default=None, ge=1)
    max_output_tokens: int = Field(default=1024, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ToolCall(Schema):
    call_id: str
    name: str
    input: dict[str, Any]
    provider_call_id: str | None = None


class ToolResult(Schema):
    call_id: str
    name: str
    content: list[ContentBlock]
    is_error: bool = False


class ModelRequest(Schema):
    actor: ActorRef
    provider: ProviderOptions
    system_text: str
    messages: list[ConversationMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)
    capability_registry: list[CapabilityRegistryEntry] = Field(default_factory=list)
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)


class ModelResponse(Schema):
    provider: ProviderName
    model: str
    content: list[ContentBlock]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason | str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    raw_response: dict[str, Any] | None = None


HookEventName = Literal[
    "before_request_interpretation",
    "after_request_interpretation",
    "before_memory_retrieval",
    "after_memory_retrieval",
    "before_context_assembly",
    "after_context_assembly",
    "before_model_call",
    "after_model_call",
    "before_tool_call",
    "after_tool_call",
    "before_skill_load",
    "after_skill_load",
    "before_mcp_call",
    "after_mcp_call",
    "before_memory_write",
    "after_memory_write",
    "after_turn",
    "on_error",
]
HookEventGroup = Literal["memory", "memory_adjacent", "runtime"]
HookHandlerKind = Literal["python", "webhook", "event_bus"]


class HookHandlerRef(Schema):
    kind: HookHandlerKind
    import_path: str | None = None
    url: str | None = None
    secret_ref: str | None = None
    topic: str | None = None


class HookEvent(Schema):
    event_id: str
    name: HookEventName
    actor: ActorRef
    turn_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class HookDefinition(Schema):
    name: HookEventName
    group: HookEventGroup
    enabled: bool = True
    handler: HookHandlerRef
    can_veto: bool = False
    can_mutate_payload: bool = False


class HookResult(Schema):
    allow: bool = True
    payload_patch: dict[str, Any] = Field(default_factory=dict)
    audit_note: str | None = None
```

Phase note: Phase 1 only needs the memory hook group: `before_memory_retrieval`, `after_memory_retrieval`, `before_memory_write`, and `after_memory_write`. The full enum is listed so later phases do not need to rename events, but implementations should enable events by phase.

Protocol note: hook contracts are protocol-adjacent because the same `HookEvent` and `HookResult` may be dispatched through local Python callbacks, an in-process event bus, HTTP webhooks, queue messages, or test harness calls. Keep transport-specific delivery fields such as HTTP headers, delivery attempts, queue offsets, acknowledgement ids, and timeout policies outside `HookEvent`; use a protocol-specific envelope or dispatcher adapter. `HookHandlerRef` makes the handler kind explicit so Python import paths, webhook targets, and event-bus topics are not hidden inside one opaque string.

Provider note: `ProviderOptions`, `ToolCall`, `ToolResult`, `ModelRequest`, `ModelResponse`, `HookEvent`, and `HookResult` are provider-neutral contracts. `ProviderName` is an open string checked by `ProviderRegistry`, while `KnownProviderName` is only a convenience for common adapters. Provider adapters should normalize Anthropic `tool_use`, OpenAI tool calls, local model JSON calls, and provider-specific stop reasons into these neutral shapes. Adapter-specific escape hatches belong in adapter contracts such as `AnthropicRuntimeOptions`, not in neutral `ProviderOptions`.

### 8.12. Anthropic-specific adapter contracts

```python
class AnthropicSkillConfig(Schema):
    type: Literal["anthropic", "custom"] = "custom"
    skill_id: str
    name: str | None = None
    version: str | None = "latest"
    enabled: bool = True


class AnthropicMcpServerConfig(Schema):
    type: Literal["url"] = "url"
    name: str
    url: str
    authorization_token_ref: str | None = None


class AnthropicMcpToolsetConfig(Schema):
    type: Literal["mcp_toolset"] = "mcp_toolset"
    mcp_server_name: str
    default_config: dict[str, Any] = Field(default_factory=dict)
    configs: dict[str, Any] = Field(default_factory=dict)


class AnthropicRuntimeOptions(Schema):
    container_id: str | None = None
    skills: list[AnthropicSkillConfig] = Field(default_factory=list)
    mcp_servers: list[AnthropicMcpServerConfig] = Field(default_factory=list)
    mcp_toolsets: list[AnthropicMcpToolsetConfig] = Field(default_factory=list)
    beta_headers: list[str] = Field(default_factory=list)
    use_code_execution: bool = False
    use_files_api: bool = False
```

Provider note: these `Anthropic*` classes are intentionally provider-specific adapter contracts for Anthropic Skills, MCP connector config, beta headers, Files API, and container/runtime options. They should be consumed only by the Anthropic adapter. Equivalent OpenAI-compatible, local-model, or other provider options should be added as separate adapter contracts, not folded into the neutral `ModelRequest`, `ModelResponse`, or memory schemas.

### 8.13. Turn state machine contracts

```python
from enum import StrEnum


class TurnState(StrEnum):
    START = "start"
    INTERPRET_REQUEST = "interpret_request"
    RETRIEVE_CONTEXT = "retrieve_context"
    ASSEMBLE_CONTEXT = "assemble_context"
    RENDER_MODEL_REQUEST = "render_model_request"
    CALL_MODEL = "call_model"
    HANDLE_MODEL_RESPONSE = "handle_model_response"
    EXECUTE_TOOL_CALLS = "execute_tool_calls"
    APPEND_TOOL_RESULTS = "append_tool_results"
    WRITE_MEMORY = "write_memory"
    FINALIZE = "finalize"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TurnMachineContext(Schema):
    turn_id: str
    state: TurnState = TurnState.START
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=8, ge=1)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    idempotency_key: str | None = None

    actor: ActorRef
    provider: ProviderOptions
    provider_runtime_options: dict[str, Any] = Field(default_factory=dict)

    current_message: str
    interpreted_request: InterpretedRequest | None = None
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)
    assembled_context: ContextAssembly | None = None
    rendered_payload: dict[str, Any] | None = None

    model_response: ModelResponse | None = None
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)

    final_text: str | None = None
    errors: list[str] = Field(default_factory=list)
```

Persistence note: `TurnMachineContext` is the primary persisted turn-execution aggregate. `TurnRecord` in Section 8.14 is a read projection that can be derived from `TurnMachineContext` plus `StateTransitionRecord` and `AuditEvent`. If `TurnRecord` is physically stored for query speed, update it transactionally with the primary context and transition log.

Provider note: the persisted turn state machine should be provider-neutral. `RENDER_MODEL_REQUEST` means "render the current neutral context through the selected provider adapter"; it does not imply Anthropic. Provider-specific render steps and runtime options should stay behind adapter-specific fields or the generic `provider_runtime_options` mapping. Do not make Anthropic content blocks, beta headers, MCP connector payloads, or container settings required for non-Anthropic providers.

State note: `TurnState` is the internal executor state. `TurnStatus` in Section 8.14 is the coarser external lifecycle status used by UI, API responses, and reports. `FINALIZE`, `FAILED`, and `CANCELLED` are terminal executor states; they map to external statuses `completed`, `failed`, and `cancelled` respectively. Resume/continuation operations should reject terminal states unless the caller explicitly creates a new replacement turn.

### 8.14. Turn, audit, and memory-write contracts

```python
TurnStatus = Literal["received", "running", "completed", "failed", "cancelled"]
RetrievalNeedKind = Literal[
    "profile",
    "conversation",
    "summary",
    "knowledge",
    "episode",
    "procedure",
    "mcp_resource",
]
MemoryWriteKind = Literal["profile", "summary", "knowledge", "episode", "procedure", "cache"]
MemoryWriteAction = Literal["created", "updated", "superseded", "skipped", "rejected"]
AuditEventType = Literal[
    "turn_received",
    "state_transition",
    "request_interpreted",
    "memory_retrieved",
    "context_assembled",
    "context_item_injected",
    "model_called",
    "tool_called",
    "skill_loaded",
    "mcp_called",
    "memory_write_proposed",
    "memory_write_committed",
    "memory_write_rejected",
    "turn_completed",
    "turn_failed",
    "turn_cancelled",
    "error",
]


class RetrievalNeed(Schema):
    kind: RetrievalNeedKind
    query: str | None = None
    reason: str
    required: bool = False
    limit: int = Field(default=5, ge=1, le=50)


class InterpretedRequest(Schema):
    turn_id: str
    task_type: str | None = None
    entities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    retrieval_needs: list[RetrievalNeed] = Field(default_factory=list)
    capability_needs: list[str] = Field(default_factory=list)
    should_write_memory: bool = True
    created_at: datetime


class RetrievalPlan(Schema):
    turn_id: str
    needs: list[RetrievalNeed] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    context_budget_tokens: int | None = Field(default=None, ge=1)
    include_recent_messages: bool = True
    include_summaries: bool = True
    created_at: datetime


class ContextAssembly(Schema):
    assembly_id: str
    turn_id: str
    actor: ActorRef
    instructions: list[str] = Field(default_factory=list)
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    context_items: list[RetrievedContextItem] = Field(default_factory=list)
    tool_definitions: list[ToolDefinition] = Field(default_factory=list)
    capability_registry: list[CapabilityRegistryEntry] = Field(default_factory=list)
    render_order: list[Literal[
        "instructions",
        "context_items",
        "capability_registry",
        "tool_definitions",
        "recent_messages",
        "current_message",
    ]] = Field(
        default_factory=lambda: [
            "instructions",
            "context_items",
            "capability_registry",
            "tool_definitions",
            "recent_messages",
            "current_message",
        ],
    )
    token_budget: int | None = Field(default=None, ge=1)
    estimated_tokens: int | None = Field(default=None, ge=0)
    omitted_item_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class StateTransitionRecord(Schema):
    transition_id: str
    turn_id: str
    from_state: TurnState | None = None
    to_state: TurnState
    reason: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnRecord(Schema):
    turn_id: str
    actor: ActorRef
    status: TurnStatus
    state: TurnState
    message_ids: list[str] = Field(default_factory=list)
    state_transition_ids: list[str] = Field(default_factory=list)
    retrieved_context_ids: list[str] = Field(default_factory=list)
    capability_call_ids: list[str] = Field(default_factory=list)
    memory_write_ids: list[str] = Field(default_factory=list)
    started_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None
    final_text: str | None = None
    error_summary: str | None = None


class MemoryWritePayload(Schema):
    profile: ProfileMemory | None = None
    episode: EpisodeRecord | None = None
    knowledge: KnowledgeDocument | None = None
    summary: ConversationSummary | None = None
    procedure: ProcedureRecord | None = None
    cache: dict[str, Any] | None = None


class MemoryWriteCandidate(Schema):
    candidate_id: str
    turn_id: str
    kind: MemoryWriteKind
    content: MemoryWritePayload
    source_refs: list[SourceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    reason: str
    requires_review: bool = False
    created_at: datetime


class CommittedMemoryWrite(Schema):
    write_id: str
    candidate_id: str | None = None
    turn_id: str
    kind: MemoryWriteKind
    action: MemoryWriteAction
    memory_id: str | None = None
    reason: str
    committed_at: datetime


class AuditPayload(Schema):
    state_transition: StateTransitionRecord | None = None
    retrieved_context_ids: list[str] = Field(default_factory=list)
    injected_context_ids: list[str] = Field(default_factory=list)
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    memory_write: CommittedMemoryWrite | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AuditEvent(Schema):
    event_id: str
    actor: ActorRef
    turn_id: str | None = None
    event_type: AuditEventType
    payload: AuditPayload = Field(default_factory=AuditPayload)
    source: SourceRef | None = None
    created_at: datetime
```

These contracts close the gap between the domain model and the runtime loop. `TurnMachineContext` is the primary persisted execution aggregate; `TurnRecord` is a durable read projection. `StateTransitionRecord` is the audit-friendly state history, `InterpretedRequest` and `RetrievalPlan` make retrieval decisions typed, `ContextAssembly` replaces unstructured assembled-context dictionaries, and `MemoryWriteCandidate` plus `CommittedMemoryWrite` model selective memory writing explicitly. `AuditEvent` is the durable explanation layer for retrieval, context injection, model calls, capability calls, errors, and memory writes.

Rendering contract: provider renderers should respect `ContextAssembly.render_order`. The default order is instructions, retrieved context, capability registry, tool definitions, recent messages, and current message. Anthropic renderers should place stable retrieved context and capability summaries in `system` when appropriate, while preserving recent messages chronologically in `messages`.

Status mapping rule: keep `TurnRecord.status` and `TurnRecord.state` synchronized. New turns start as `status="received"` with `state=TurnState.START`; active execution should use `status="running"`; `TurnState.FINALIZE` should set `status="completed"`; `TurnState.FAILED` should set `status="failed"`; and `TurnState.CANCELLED` should set `status="cancelled"`. Terminal states should set `ended_at` as the turn end timestamp.

### 8.15. Evaluation contracts

```python
EvaluationTarget = Literal[
    "retrieval",
    "ranking",
    "context_assembly",
    "memory_write",
    "state_machine",
    "provider_adapter",
]


class EvaluationCase(Schema):
    case_id: str
    target: EvaluationTarget
    actor: ActorRef | None = None
    input_payload: dict[str, Any]
    expected_payload: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class EvaluationMetricResult(Schema):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationRun(Schema):
    run_id: str
    target: EvaluationTarget
    case_ids: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "succeeded", "failed"]
    metrics: list[EvaluationMetricResult] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
```

### 8.16. Service wrapper contracts

Use wrapper contracts only at service boundaries when a call needs to aggregate several Section 8 contracts. These wrappers are not HTTP-specific; HTTP headers, status codes, WebSocket ids, queue offsets, and retry metadata belong in transport envelopes.

```python
class RunTurnRequest(Schema):
    actor: ActorRef
    current_message: str
    provider: ProviderOptions
    provider_runtime_options: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    idempotency_key: str | None = None
    enable_memory_writes: bool = True


class RunTurnResult(Schema):
    turn: TurnRecord
    context: TurnMachineContext
    response: ModelResponse | None = None
    memory_writes: list[CommittedMemoryWrite] = Field(default_factory=list)
    audit_event_ids: list[str] = Field(default_factory=list)


class ResumeTurnRequest(Schema):
    turn_id: str
    actor: ActorRef
    idempotency_key: str | None = None


class SubmitToolResultsRequest(Schema):
    turn_id: str
    actor: ActorRef
    tool_results: list[ToolResult]
    idempotency_key: str | None = None


class SubmitApprovalRequest(Schema):
    turn_id: str
    actor: ActorRef
    approved: bool
    reason: str | None = None
    idempotency_key: str | None = None


class AssembleContextRequest(Schema):
    turn_id: str
    actor: ActorRef
    retrieval_plan: RetrievalPlan
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)
    token_budget: int | None = Field(default=None, ge=1)


class SearchMemoryRequest(Schema):
    actor: ActorRef
    query: str
    kinds: list[RetrievedContextKind] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)


class MemoryCommitRequest(Schema):
    actor: ActorRef
    candidates: list[MemoryWriteCandidate]
    mode: Literal["auto", "reviewed", "dry_run"] = "auto"
    idempotency_key: str | None = None


class ConsolidationRequest(Schema):
    actor: ActorRef
    memory_kinds: list[MemoryWriteKind] = Field(default_factory=list)
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    dry_run: bool = True


class McpReadResourceRequest(Schema):
    actor: ActorRef
    server_id: str
    uri: str
```

## 9. FastAPI/HTTP Adapter for the Python + Anthropic Baseline

This section is a concrete optional transport adapter for the baseline in Section 4 and the Python implementation target in Section 5. It is not the source of truth for the memory system. Sections 1-2 define the provider-, protocol-, and language-neutral design; Section 3 defines the runtime behavior; Sections 6-7 define model-provider boundaries; and Section 8 defines the Python data contracts.

The HTTP adapter may depend on FastAPI, JSON, Pydantic serialization, and the Anthropic adapter because this project has selected Python and Anthropic as the first implementation target. That dependency is acceptable only at the transport and provider-adapter boundary. Core memory behavior should remain in `services/`, not in route handlers.

### 9.1. Purpose and boundaries

Use the HTTP adapter when an external UI, local desktop shell, browser extension, automation process, or another service needs to call the agentic-memory runtime out of process.

The HTTP adapter should:

- Translate HTTP requests into Section 8 Pydantic contracts.
- Call service-layer operations from Section 5.2, especially `TurnRunner`, `TurnStateMachine`, `RetrievalService`, `ContextManager`, `ModelGateway`, `MemoryWriter`, `HookManager`, and `AuditLogger`.
- Serialize service results back to JSON without inventing a second schema system.
- Keep authentication, request ids, idempotency keys, rate limits, and transport errors at the HTTP boundary.
- Keep memory policy, retrieval policy, state transitions, provider selection, memory writes, and audit semantics inside `services/`.

The HTTP adapter should not:

- Become the canonical API design for the system.
- Store memory directly from route handlers.
- Execute tools directly from route handlers.
- Encode Anthropic wire shapes into neutral memory endpoints.
- Require HTTP for local desktop use, tests, workers, or in-process integrations.

### 9.2. Route-to-service mapping rule

Every route should be a thin mapping from HTTP to a logical service call:

```text
HTTP request
  -> FastAPI route
  -> Pydantic request contract
  -> service call
  -> Pydantic response contract
  -> HTTP response or stream event
```

Recommended route implementation pattern:

```python
@router.post('/turns')
async def run_turn(
    request: RunTurnRequest,
    runtime: AgentMemoryRuntime = Depends(get_runtime),
) -> RunTurnResult:
    return await runtime.run_turn(request)
```

Keep the route layer responsible for transport concerns only: authentication, authorization claims, request validation, idempotency headers, timeout policy, HTTP status codes, response serialization, streaming setup, and error translation.

### 9.3. Minimal HTTP routes when added early

HTTP is not a Phase 1 deliverable for the local-first baseline. If an HTTP adapter ships before Phase 3, it should expose only the Phase 1 feature set: a single-pass memory-aware turn, persisted state inspection, retrieval, memory writes, and audit. These routes should use Section 8 contracts or thin request/response wrappers around those contracts.

| Route | Service operation | Request body | Purpose | Section 8 contracts |
|---|---|---|---|---|
| `POST /turns` | `RunTurn` | `RunTurnRequest` | Run the core memory path: interpret, retrieve, assemble context, call model once, write memory, audit. | `RunTurnResult`, `TurnMachineContext`, `TurnRecord`, `ModelResponse` |
| `GET /turns/{turn_id}` | `GetTurnState` | none | Return current turn status, internal state, final text, errors, retrieved context ids, and memory-write ids. | `TurnRecord`, `TurnMachineContext` |
| `POST /turns/{turn_id}/resume` | `ResumeTurn` | `ResumeTurnRequest` | Resume a non-terminal turn after retryable interruption or crash. Do not resume `completed`, `failed`, or `cancelled` turns in place. | `TurnMachineContext`, `StateTransitionRecord` |
| `GET /turns/{turn_id}/audit` | `WriteAuditEvent` query side | none | Return audit events for a turn. | `AuditEvent` |
| `POST /memory/search` | `SearchProfileMemory`, `SearchKnowledge`, `SearchEpisodes`, `SearchProcedures` | `SearchMemoryRequest` | Retrieve profile, knowledge, episode, and procedure memories through the retrieval service. | `RetrievedContextItem`, source memory records |
| `PATCH /memory/profile` | `PatchProfileMemory` | `MemoryCommitRequest` | Commit durable profile-memory changes through the memory writer. | `MemoryWriteCandidate`, `CommittedMemoryWrite`, `ProfileMemory` |
| `POST /knowledge` | `UpsertKnowledge` | operation-specific wrapper around `KnowledgeDocument` or `KnowledgeChunk` | Store documents or chunks as retrievable knowledge and optionally index them. | `KnowledgeDocument`, `KnowledgeChunk`, `VectorUpsertRequest` |
| `POST /context/assemble` | `AssembleContext` | `AssembleContextRequest` | Build provider-neutral context from retrieved memory and current turn data. | `ContextAssembly`, `RetrievedContextItem` |
| `POST /model/call` | `CallModel` | `ModelRequest` | Optional diagnostic route for calling the selected provider with a neutral request. | `ModelRequest`, `ModelResponse` |

For a local-first desktop implementation, even these routes are optional. The same service calls can be used in-process without HTTP.

### 9.4. Optional streaming routes

Streaming is a transport feature, not a memory-system requirement. Add it only when the user interface needs live turn progress, token output, tool progress, approval prompts, cancellation acknowledgement, or state-transition events.

Recommended options:

- `GET /turns/{turn_id}/events`: SSE stream for one-way server-to-client events.
- `WS /turns/{turn_id}`: WebSocket for bidirectional interactive sessions where the client may send cancellation, approval, or steering events during a running turn.
- `POST /turns/{turn_id}/cancel`: non-streaming cancellation route that transitions a non-terminal turn to `CANCELLED`.

Suggested event names:

```text
turn.received
turn.state_changed
memory.retrieved
context.assembled
model.started
model.delta
model.completed
capability.requested
capability.completed
memory.write_proposed
memory.write_committed
turn.completed
turn.failed
turn.cancelled
```

The persisted turn store remains authoritative. SSE and WebSocket events are live projections of state changes, not the source of state.

### 9.5. Anthropic-specific HTTP surface

Anthropic-specific routes are optional diagnostics and adapter controls. They should not be required for normal memory operation because `POST /turns` should call the selected provider through `ModelGateway`.

These routes may bypass `services/` and call `integrations/anthropic/` directly. Treat them as diagnostic or adapter-inspection routes; production callers should use `/turns` or the in-process `AgentMemoryRuntime`.

Acceptable Anthropic-specific routes:

| Route | Purpose | Boundary rule |
|---|---|---|
| `POST /anthropic/messages/render` | Render a neutral `ModelRequest` plus `AnthropicRuntimeOptions` into an Anthropic Messages payload for debugging. | Return rendered payload as adapter output; do not persist memory here. |
| `POST /anthropic/messages/call` | Call Anthropic directly through the adapter for diagnostics. | Normalize the result into `ModelResponse`; production turns should still use `ModelGateway`. |
| `POST /anthropic/tools/render` | Render neutral `ToolDefinition` records into Anthropic tool schema. | Do not execute tools here. |
| `POST /anthropic/skills/render` | Render selected local procedures or skills into Anthropic Skills configuration when enabled. | Local procedure/skill registry remains source of truth. |
| `POST /anthropic/mcp/render` | Render neutral MCP config into Anthropic MCP connector payloads when enabled. | Local MCP registry remains source of truth. |

Keep Anthropic fields under explicit adapter options, for example:

```json
{
  "provider": {
    "name": "anthropic",
    "model": "claude-sonnet-4-5",
    "max_output_tokens": 1200
  },
  "provider_runtime_options": {
    "anthropic": {
      "container_id": "container_agentic_memory",
      "skills": [],
      "mcp_servers": [],
      "mcp_toolsets": [],
      "beta_headers": []
    }
  }
}
```

Do not put Anthropic-only fields directly on neutral memory objects such as `ProfileMemory`, `EpisodeRecord`, `KnowledgeDocument`, `ContextAssembly`, or `TurnRecord`.

### 9.6. Error handling and idempotency

HTTP errors should translate transport and service failures without hiding state-machine outcomes.

Recommended conventions:

- Use `400` for malformed request payloads or invalid route parameters.
- Use `401` or `403` for authentication and authorization failures.
- Use `404` when a tenant-scoped record is not found.
- Use `409` for stale turn-state transitions, duplicate idempotency keys, memory conflicts requiring review, or attempts to continue terminal turns.
- Use `422` for valid JSON that fails domain validation.
- Use `429` for rate limits.
- Use `500` for unexpected server errors after recording an `AuditEvent` when a turn is involved.
- Use `503` for unavailable provider, database, vector index, or external integration dependencies.

For write routes, support an `Idempotency-Key` header or request-level idempotency field. This matters for `POST /turns`, `POST /knowledge`, `PATCH /memory/profile`, `POST /memory/commit`, and cancellation requests.

### 9.7. What remains provider/protocol/language neutral

Even with a concrete FastAPI and Anthropic adapter, these parts should remain neutral:

- Domain concepts in Section 1.
- Logical operations in Section 2.
- Runtime state-machine semantics in Section 3.
- Provider-neutral LLM contracts in Section 6.
- Python data contracts that are not explicitly Anthropic-specific in Section 8.
- Memory retrieval, ranking, context assembly, hooks, memory writes, audit, and evaluation behavior in `services/`.

Section 9 is allowed to depend on HTTP, FastAPI, Python serialization, and Anthropic adapter options. It should not cause those dependencies to leak backward into Sections 1-8.

## 10. Service Interface Design

This section defines Python service boundaries that implement the logical API from Section 2 using the data contracts from Section 8. These interfaces are for the in-process runtime. The FastAPI/HTTP adapter in Section 9 should call these services instead of defining separate memory behavior.

Section 10 is implementation-facing. It may use Python `Protocol`, async methods, dependency injection, and Pydantic request wrappers. It should not introduce new domain concepts, provider wire formats, HTTP schemas, or database schemas.

### 10.1. Purpose and boundaries

Use service interfaces to make the runtime testable, replaceable, and modular:

- `services/` owns orchestration, memory policy, state transitions, ranking, context assembly, hooks, memory writes, and lifecycle behavior.
- `repositories/` owns durable storage of turns, messages, memories, summaries, episodes, procedures, audit events, and indexes.
- `llm/` owns provider-neutral model interfaces.
- `integrations/` owns provider-specific and external-system adapters.
- `api/` is optional and should only translate HTTP requests into service calls.

The service layer should depend inward on Section 8 contracts and outward on repository, model, embedding, and integration protocols. Route handlers, CLI commands, desktop UI code, workers, and tests should call the same services.

### 10.2. Interface dependency rules

- Prefer Section 8 contracts directly for method inputs and outputs.
- Use operation-specific request/response wrappers only when one service call needs to aggregate several Section 8 contracts.
- Keep wrapper types in `agentic_memory.schemas`, not inside service implementations.
- Keep wrappers provider-neutral by default. Put Anthropic-only fields under `provider_runtime_options` or Section 8.12 adapter contracts.
- Do not let Phase 1 interfaces import Phase 2 or Phase 3 modules. Phase 2 and Phase 3 services can depend on Phase 1 contracts, not the reverse.
- Keep provider adapters responsible for rendering and normalization only. The persisted state machine owns loop progression, terminal-state mapping, retries, and iteration bounds.
- Keep repository protocols storage-oriented. They should not decide what memory is important, what context should be injected, or whether a turn should advance.

### 10.3. Phase 1 service interfaces

Phase 1 implements the core memory loop: run a turn, persist state, retrieve memory, assemble context, call Anthropic through the provider-neutral gateway, write selected memory, dispatch memory-specific hooks, and audit the path. `RequestInterpreter` and `ImportanceScorer` should be rule-based first; LLM-based interpretation/scoring is a Phase 2 quality upgrade.

| Interface | Package | Main Section 2 operations | Primary Section 8 contracts |
|---|---|---|---|
| `AgentMemoryRuntime` | `services/turn_runner.py` | `RunTurn`, `ResumeTurn` | `RunTurnRequest`, `RunTurnResult`, `ResumeTurnRequest`, `TurnMachineContext` |
| `TurnStateMachine` | `services/turn_state_machine.py` | `TransitionTurnState`, `ResumeTurn` | `TurnState`, `TurnMachineContext`, `StateTransitionRecord` |
| `TurnStore` | `repositories/turns.py` | `GetTurnState` | `TurnMachineContext`, `TurnRecord`, `StateTransitionRecord` |
| `RequestInterpreter` | `services/interpreter.py` | `InterpretRequest` | `InterpretedRequest`, `RetrievalNeed`, `RetrievalPlan` |
| `RetrievalService` | `services/retrieval.py` | `SearchProfileMemory`, `SearchKnowledge`, `SearchEpisodes`, `SearchProcedures`, `SearchVectorIndex`, `RenderEpisodes` | `SearchMemoryRequest`, `RetrievedContextItem`, `ProfileMemory`, `KnowledgeChunk`, `EpisodeRecord`, `ProcedureRecord`, `VectorSearchResult` |
| `ContextManager` | `services/context_manager.py` | `EstimateContextBudget`, `AssembleContext` | `AssembleContextRequest`, `ContextAssembly`, `RetrievedContextItem`, `ProviderOptions` |
| `ModelGateway` | `services/model_gateway.py` | `RenderModelRequest`, `CallModel` | `ModelRequest`, `ModelResponse`, `ProviderOptions` |
| `ModelProvider` | `llm/base.py` | `CallModel` | `ModelRequest`, `ModelResponse` |
| `ProviderRegistry` | `providers/registry.py` | provider selection | `ProviderName`, `ProviderOptions` |
| `VectorIndex` | `repositories/vector_index.py` | `UpsertVectorIndex`, `SearchVectorIndex` | `VectorUpsertRequest`, `VectorUpsertResult`, `VectorSearchRequest`, `VectorSearchResult` |
| `EmbeddingProvider` | `embeddings/base.py` | `UpsertVectorIndex`, `SearchVectorIndex` | `embedding_model`, `vector_dim`, embedding vectors |
| `MemoryWriter` | `services/memory_writer.py` | `ProposeMemoryWrites`, `CommitMemoryWrites`, `PatchProfileMemory`, `CreateEpisode`, `UpsertKnowledge` | `MemoryWriteCandidate`, `MemoryCommitRequest`, `CommittedMemoryWrite` |
| `MemoryRanker` | `services/memory_ranker.py` | `RankMemory` | `MemoryRankingPolicy`, `MemoryRankedItem`, `MemoryScoreComponents` |
| `ImportanceScorer` | `services/importance_scorer.py` | `ScoreMemoryImportance` | `MemoryScoreComponents`, `MemoryWriteCandidate` |
| `HookManager` | `services/hook_manager.py` | `DispatchHook` | `HookEvent`, `HookDefinition`, `HookResult` |
| `AuditLogger` | `services/audit_logger.py` | `WriteAuditEvent` | `AuditEvent`, `StateTransitionRecord` |

Phase 1 adapters:

- `AnthropicMessagesAdapter` in `integrations/anthropic/` implements the neutral `ModelProvider` contract for Anthropic Messages request rendering, model calls, and response normalization.
- Local vector/embedding adapters implement `VectorIndex` and `EmbeddingProvider`; only one concrete vector backend is required to run Phase 1.

Representative Phase 1 protocols:

```python
from typing import Protocol


class AgentMemoryRuntime(Protocol):
    async def run_turn(self, request: RunTurnRequest) -> RunTurnResult: ...
    async def resume_turn(self, request: ResumeTurnRequest) -> RunTurnResult: ...


class RequestInterpreter(Protocol):
    async def interpret(self, context: TurnMachineContext) -> InterpretedRequest: ...


class TurnStateMachine(Protocol):
    async def run(self, context: TurnMachineContext) -> TurnMachineContext: ...
    async def step(self, context: TurnMachineContext) -> TurnMachineContext: ...
    async def cancel(self, turn_id: str, reason: str) -> TurnRecord: ...


class ProviderRegistry(Protocol):
    def register(self, provider: ModelProvider) -> None: ...
    def get_provider(self, name: ProviderName) -> ModelProvider: ...


class EmbeddingProvider(Protocol):
    embedding_model: str
    vector_dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class RetrievalService(Protocol):
    async def retrieve(self, request: SearchMemoryRequest) -> list[RetrievedContextItem]: ...


class ContextManager(Protocol):
    async def assemble(self, request: AssembleContextRequest) -> ContextAssembly: ...


class ModelGateway(Protocol):
    async def call(self, request: ModelRequest) -> ModelResponse: ...


class MemoryWriter(Protocol):
    async def propose(self, context: TurnMachineContext) -> list[MemoryWriteCandidate]: ...
    async def commit(self, request: MemoryCommitRequest) -> list[CommittedMemoryWrite]: ...
```

### 10.4. Phase 2 service interfaces

Phase 2 adds quality and local capability support: summaries, lifecycle management, conflicts, consolidation, local model-visible tools, local skill packaging, expanded hooks, evaluation, reranking, and background jobs.

| Interface | Package | Main Section 2 operations | Primary Section 8 contracts |
|---|---|---|---|
| `Summarizer` | `services/summarizer.py` | `SummarizeConversation`, `IsCompressionNeeded` | `ConversationSummary`, `EpisodeNote`, `ConversationMessage` |
| `MemoryLifecycleManager` | `services/memory_lifecycle.py` | `CheckMemoryConflicts`, `ConsolidateMemory` | `MemoryConflict`, `ConflictCheckResult`, `ConsolidationRun`, `MemoryMetadata` |
| `Reranker` | `services/reranker.py` | retrieval-quality reranking | `RetrievedContextItem`, `MemoryRankedItem` |
| `CapabilityRegistry` | `services/capability_registry.py` | `ListCapabilities`, `RenderCapabilityRegistry` | `CapabilityRegistryEntry`, `ToolDefinition`, `ContextAssembly` |
| `CapabilityLoop` | `services/capability_loop.py` | `SubmitToolResults`, `SubmitApproval`, `ExecuteToolCall` | `TurnMachineContext`, `ToolCall`, `ToolResult`, `ModelResponse` |
| `ToolRunner` | `services/tool_runner.py` | `ExecuteToolCall` | `ToolCall`, `ToolResult`, `ToolDefinition` |
| `SkillManager` | `services/skill_manager.py` | `LoadSkill`, `SearchProcedures` | `ProcedureRecord`, `CapabilityRegistryEntry`, `RetrievedContextItem` |
| `EvaluationService` | `services/evaluation.py` | `RunEvaluation` | `EvaluationCase`, `EvaluationMetricResult`, `EvaluationRun` |

Representative Phase 2 protocols:

```python
class ToolRunner(Protocol):
    async def execute(self, call: ToolCall) -> ToolResult: ...


class CapabilityLoop(Protocol):
    async def submit_tool_results(
        self,
        request: SubmitToolResultsRequest,
    ) -> TurnMachineContext: ...
    async def submit_approval(
        self,
        request: SubmitApprovalRequest,
    ) -> TurnMachineContext: ...


class MemoryLifecycleManager(Protocol):
    async def check_conflicts(
        self,
        candidate: MemoryWriteCandidate,
    ) -> ConflictCheckResult: ...
    async def consolidate(self, request: ConsolidationRequest) -> ConsolidationRun: ...
```

### 10.5. Phase 3 service interfaces

Phase 3 adds external integrations and provider-native capability rendering: MCP, Anthropic-native Skills, provider-native/server-side tools, Claude Code compatibility, and optional external transport adapters.

| Interface | Package | Main Section 2 operations | Primary Section 8 contracts |
|---|---|---|---|
| `McpManager` | `services/mcp_manager.py` | `ListMcpServers`, `ReadMcpResource`, `RenderMcpForProvider` | `McpReadResourceRequest`, `McpServerRecord`, `McpResource`, `McpResourceTemplate`, `RetrievedContextItem`, `AnthropicMcpServerConfig` |
| `AnthropicSkillRenderer` | `integrations/anthropic/skills.py` | `LoadSkill`, provider-native skill rendering | `ProcedureRecord`, `CapabilityRegistryEntry`, `AnthropicSkillConfig` |
| `AnthropicMcpRenderer` | `integrations/anthropic/mcp_connector.py` | `RenderMcpForProvider` | `McpServerRecord`, `AnthropicMcpServerConfig`, `AnthropicMcpToolsetConfig` |
| `ExternalTransportAdapter` | `api/` or integration-specific package | Optional HTTP, SSE, WebSocket, queue, or other transport mapping | Section 8 contracts plus transport envelopes |

Representative Phase 3 protocols:

```python
class McpManager(Protocol):
    async def list_servers(self, actor: ActorRef) -> list[McpServerRecord]: ...
    async def read_resource(self, request: McpReadResourceRequest) -> RetrievedContextItem: ...


class AnthropicMcpRenderer(Protocol):
    async def render(
        self,
        servers: list[McpServerRecord],
    ) -> AnthropicRuntimeOptions: ...
```

### 10.6. Mapping to Sections 1-9

| Earlier section | Relationship to Section 10 |
|---|---|
| Section 1 | Defines the concepts that service interfaces manipulate. |
| Section 2 | Defines logical operations; Section 10 groups those operations into Python service boundaries. |
| Section 3 | Defines runtime behavior; Section 10 provides the service interfaces that implement it. |
| Section 4 | Defines local-first deployment assumptions; Section 10 keeps HTTP optional and supports in-process runtime calls. |
| Section 5 | Defines package layout; Section 10 maps interfaces to `services/`, `repositories/`, `llm/`, `embeddings/`, and `integrations/`. |
| Section 6 | Defines provider-neutral LLM architecture; Section 10 represents it with `ModelGateway`, `ModelProvider`, and `ProviderRegistry`. |
| Section 7 | Defines Anthropic adapter mapping; Section 10 keeps Anthropic-specific interfaces in adapter lists and `integrations/anthropic/`. |
| Section 8 | Defines data contracts; Section 10 consumes those contracts but does not redefine them. |
| Section 9 | Defines optional HTTP routes; Section 10 is the service layer those routes call. |

### 10.7. Request and response wrapper rules

Wrapper classes such as `RunTurnRequest`, `RunTurnResult`, `ResumeTurnRequest`, `AssembleContextRequest`, `SearchMemoryRequest`, `MemoryCommitRequest`, or `ConsolidationRequest` are allowed when a service operation needs a stable call boundary. They should follow these rules:

- A wrapper should aggregate Section 8 contracts; it should not create a parallel domain model.
- A wrapper should be provider-neutral unless its name is explicitly provider-specific, such as `AnthropicRenderRequest`.
- A wrapper should not contain HTTP headers, status codes, WebSocket ids, queue offsets, or transport retry metadata. Put those in Section 9 transport envelopes.
- A wrapper should not contain SQLAlchemy models or database sessions. Keep storage details inside repositories.
- A wrapper should include `idempotency_key` for side-effecting operations that may be retried.
- A wrapper should be versioned when external callers depend on it, but internal-only wrappers can evolve with the implementation.

## 11. Protocol Choices

The API design is mostly protocol-agnostic. The logical API defines operations and data shapes: `run turn`, `retrieve memory`, `rank memories`, `execute tool`, `append tool result`, and `commit memory`. HTTP, WebSocket, gRPC, local function calls, message queues, and MCP are transport or integration choices.

### 11.1. Protocol options

- **HTTP/HTTPS REST**: best default for public APIs, admin APIs, CRUD-like memory operations, retrieval, ranking, audit logs, and configuration. It is simple, debuggable, cacheable, and broadly compatible.
- **Server-Sent Events (SSE)**: useful for one-way streaming when the client mostly receives model output, state changes, and progress events.
- **WebSocket**: useful for interactive agent sessions where client and server both send events, such as partial model output, tool progress, cancellation, human approval, hook events, and resume signals.
- **gRPC**: useful for internal service-to-service calls, high throughput, typed contracts, and lower overhead. It is less convenient directly from browsers.
- **Message queues / event buses**: useful for async memory writes, consolidation jobs, embedding jobs, audit logging, delayed tool execution, and retries.
- **Local in-process calls**: useful for local agent runtimes where API boundaries are still helpful but network overhead is unnecessary.
- **stdio / subprocess protocol**: useful for local tools, CLI agents, MCP servers, and plugin-style integrations.
- **MCP**: useful as a capability/resource protocol between the agent runtime and external tools or resources. It should not be treated as the main public app API.

### 11.2. Recommended choices by deployment

For a local agent runtime with a remote LLM:

- This is the best fit for Codex- or Claude Code-style agents.
- Keep the workspace, shell, files, git repository, tools, local skills, local MCP servers, hooks, and turn state machine in the local runtime.
- Use HTTPS from the local runtime to the remote LLM provider.
- Use local function calls, stdio/subprocess, or MCP for local tools and resources.
- Use an embedded DB plus local vector index when memory should stay local; use a remote memory service only when cross-device continuity or team sharing is required.
- Use an internal event bus, local WebSocket, or local HTTP streaming only when the UI needs live turn updates or user intervention.
- Use local jobs for embeddings, memory consolidation, stale-memory cleanup, vector reindexing, and optional sync.

For a local agent runtime with a local LLM:

- Use direct Python calls, local HTTP, gRPC, or Unix socket between the agent runtime and local model server.
- Keep workspace access, tools, skills, MCP, memory retrieval, vector search, hooks, and state-machine persistence local.
- Use an embedded DB plus local vector index for profile memory, episodes, documents, audit records, and state-machine state.
- Use stdio/subprocess or MCP for local tools, local files, and plugin-style integrations.
- Use an internal event bus or local WebSocket only when the UI needs live turn updates or user intervention.
- Use local queues or scheduled jobs for memory lifecycle work and periodic export/sync.

For a cloud agent runtime with a remote LLM:

- Use this when the agent operates on cloud resources, SaaS tools, shared workspaces, team approvals, or sandboxed remote dev environments.
- Use HTTPS REST for browser/backend operations such as memory search, session state, configuration, and audit views.
- Use HTTPS from the cloud runtime to the remote LLM provider.
- Keep provider keys, remote tools, SaaS API credentials, memory retrieval, MCP access, hooks, and memory writes in the cloud backend.
- Use SSE when the browser only observes live turns; use WebSocket when users approve, cancel, or steer tool execution.
- Use queues/event buses for embeddings, importance scoring, consolidation, vector reindexing, audit export, and post-turn memory writes.
- Use object storage plus signed URLs for large documents, generated artifacts, trace exports, and uploaded files.

For a hybrid local runtime with remote memory:

- Keep tools, skills, local MCP servers, shell, files, and the state machine in the local runtime.
- Use HTTPS or gRPC from the local runtime to a remote memory service such as Postgres/pgvector, a vector service, or a dedicated memory API.
- Use a local cache for recent turns, active project context, and retryable writes so the agent can tolerate network loss.
- Use a sync protocol or durable local queue for memory writes, embeddings, and audit events that must eventually reach the remote service.
- Use access-control checks on both sides: local runtime for workspace access, remote memory service for tenant/user/project scoping.
- Use local artifact paths for local-only files and object storage/signed URLs for artifacts that need cross-device or team access.

For backend-to-backend agentic memory services:

- Use REST/OpenAPI for public or cross-team APIs where debuggability and compatibility matter.
- Use gRPC for high-throughput internal services such as vector search, ranking, embedding, graph retrieval, and context assembly when REST overhead becomes measurable.
- Use queues/event buses for durable asynchronous workflows.
- Use webhooks when third-party systems need to notify the memory layer about ticket updates, document changes, policy changes, or completed workflows.
- Use MCP for tool/resource integration when the agent needs external capabilities, documents, prompts, or resource templates.
- Use a persisted state-machine store for long-running turns regardless of transport. WebSocket should carry live events, not be the only source of turn state.

### 11.3. Recommended choices by workflow requirement

- **Live observation only**: use SSE when the client only needs to receive model tokens, state changes, retrieved-memory summaries, and tool progress.
- **Live intervention**: use WebSocket or a local event bus when the client can cancel, approve/reject tool calls, reject memory writes, revise instructions, or redirect retrieval during a turn.
- **Durable background work**: use a queue or event bus for embeddings, importance scoring, memory consolidation, stale-memory cleanup, audit export, and vector reindexing.
- **Tool/resource integration**: use MCP for external tools, resource templates, prompts, and connected systems; use local function calls when tools live in the same process.
- **Large artifacts**: use object storage, file handles, or signed URLs for documents, generated reports, trace exports, and uploaded files.
- **High-throughput internal calls**: use gRPC, Unix sockets, or in-process calls for vector search, embedding, graph traversal, ranking, and context assembly when latency or throughput requires it.
- **Strict audit/replay**: use request/response APIs plus structured event logs and persisted turn-state records; do not rely on transient streaming events as the audit source.
- **Offline/local-first mode**: use an embedded DB, local vector index, local queue, and periodic sync protocol rather than assuming continuous network access.

### 11.4. Practicality of mixing protocols

Mixing protocols is practical only when each protocol solves a concrete problem that the simpler baseline does not. Otherwise it adds maintenance and runtime complexity without enough benefit.

Complexity it adds:

- More failure modes: WebSocket reconnects, queue retries, MCP process crashes, provider HTTPS errors, and local model-server failures.
- More auth paths: browser auth, local tool permissions, provider keys, remote memory credentials, and MCP server access.
- More observability work: logs and traces must correlate REST calls, streams, queue jobs, tool calls, state transitions, hooks, and memory writes.
- More schemas and adapters: the same logical operation may need mappings for HTTP, local calls, MCP, queue events, and provider-specific payloads.
- More deployment burden: queues, workers, local daemons, MCP servers, vector databases, remote databases, and sync jobs all need lifecycle management.
- More testing: unit tests for logical services plus integration tests for protocol adapters, retries, cancellation, and recovery.

When the benefit is worth it:

- **HTTPS to a remote LLM** is necessary when using providers such as Anthropic or OpenAI.
- **stdio, MCP, or local calls for tools** are worth it when the agent needs local shell, files, git, apps, or plugin-style resources.
- **A queue or local job runner** is worth it when embeddings, consolidation, memory writes, audit export, or sync must survive crashes and retries.
- **SSE or WebSocket** is worth it only when users need live tokens, state transitions, tool progress, approval, cancellation, or steering.
- **A remote memory database or service** is worth it when cross-device continuity, team sharing, centralized backup, or audit governance matters.
- **gRPC** is worth it only when REST, local HTTP, or in-process calls become a measurable bottleneck.

### 11.5. What protocol choice influences

- **Latency**: WebSocket, gRPC, Unix sockets, and local calls reduce repeated connection overhead. Streaming improves perceived latency even when total runtime is unchanged.
- **Reliability**: HTTP is simple and easy to retry. Queues are better for durable background work. WebSocket requires reconnect, resume, and idempotency logic.
- **Security**: HTTPS is the default for remote APIs. Local protocols reduce exposure but still need local access control. WebSocket needs authentication, authorization, and session revocation.
- **Streaming UX**: SSE and WebSocket matter for live tokens, tool progress, user cancellation, and human-in-the-loop approvals.
- **Operational complexity**: REST is easiest to operate. WebSocket, gRPC, queues, and MCP add more moving parts.
- **Auditability**: request/response protocols are easier to log. Streaming and event protocols need structured event logs.
- **Compatibility**: browsers prefer HTTPS, SSE, and WebSocket. Internal services may prefer gRPC. Local tools often use stdio or MCP.

## 12. Implementation Sequence

Use this as the build order. Sections 1-3 define the concepts and phases; Section 5.3 defines the package layout by phase. This section only describes the practical implementation sequence.

1. Define the provider-neutral schemas, repository interfaces, `ModelProvider`, `ProviderRegistry`, and Anthropic Messages adapter for a single model call.
2. Implement the persisted turn state machine with the core memory-path states, state-transition audit events, resumable `TurnMachineContext`, and terminal mapping into `TurnRecord.status` for `FINALIZE`, `FAILED`, and `CANCELLED`.
3. Implement local storage for sessions, messages, recent conversation history, profile memory, simple episodes, local procedures, knowledge items, memory writes, and audit events.
4. Add embedding and vector-index adapters for basic semantic retrieval, starting with SQLite/local vector storage or ChromaDB for development and local PostgreSQL/pgvector for production-like local deployments.
5. Implement request interpretation, retrieval planning, profile/episode/procedure/knowledge retrieval, basic ranking, and context assembly.
6. Implement memory-specific hooks: `before_memory_retrieval`, `after_memory_retrieval`, `before_memory_write`, and `after_memory_write`.
7. Implement provider request rendering, model call execution, response capture, selective memory-write proposal, importance scoring, commit logic, and audit logging.
8. Add conversation summaries, context compression, improved retrieval scoring, reranking, recency decay, lifecycle policy, conflict checks, consolidation, and evaluation runs.
9. Add local model-visible tools, provider-neutral `ToolRunner`, `ToolCall` and `ToolResult` handling, capability-loop states, iteration limits, and approval/cancellation paths.
10. Add a local skill registry, richer procedural-memory packaging, executable skill assets, and expanded memory-adjacent/runtime hooks.
11. Add MCP registry, local MCP client wrappers, MCP fallback tools, and provider-native MCP connectors only when external resources justify the complexity.
12. Add Anthropic Skills rendering, provider-native/server-side tools, advanced transports, queues, distributed workers, or remote memory services only when deployment requirements justify them.

## 13. Anthropic references

- [Messages API](https://docs.anthropic.com/en/api/messages)
- [Tool use with Claude](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)
- [MCP connector](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector)
- [Using Agent Skills with the API](https://docs.claude.com/en/api/skills-guide)
- [Claude Code hooks reference](https://docs.claude.com/en/docs/claude-code/hooks)
