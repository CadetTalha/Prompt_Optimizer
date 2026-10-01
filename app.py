import os

import streamlit as st
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Project Prompt Optimizer",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# LOAD OPENROUTER API KEY
# ============================================================

if "OPENROUTER_API_KEY" not in st.secrets:
    st.error(
        "OPENROUTER_API_KEY is not configured. "
        "Please add it in Streamlit Community Cloud Secrets."
    )
    st.stop()

os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openrouter/free"

TEMPERATURE_MAGIC = 0.2
TEMPERATURE_INSTRUCTIONS = 0.2

MAX_TOKENS_MAGIC = 5000
MAX_TOKENS_INSTRUCTIONS = 5000


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🧠 AI Project Prompt Optimizer")

st.markdown(
    """
Transform a raw project prompt into:

- **Magic Prompt** — a deeply engineered project specification.
- **Project Instructions** — persistent instructions suitable for
  an AI Project's instruction section.
"""
)

st.divider()


# ============================================================
# LLM CLIENTS
# ============================================================

@st.cache_resource
def get_magic_llm():

    return ChatOpenRouter(
        model=MODEL,
        temperature=TEMPERATURE_MAGIC,
        max_tokens=MAX_TOKENS_MAGIC,
        max_retries=0,
    )


@st.cache_resource
def get_instructions_llm():

    return ChatOpenRouter(
        model=MODEL,
        temperature=TEMPERATURE_INSTRUCTIONS,
        max_tokens=MAX_TOKENS_INSTRUCTIONS,
        max_retries=0,
    )


magic_llm = get_magic_llm()
instructions_llm = get_instructions_llm()


# ============================================================
# MAGIC PROMPT SYSTEM PROMPT
# ============================================================

MAGIC_PROMPT_SYSTEM = """
You are a senior AI prompt architect and software project specification engineer.

Your task is to transform the user's raw project prompt into a high-quality "Magic Prompt" that another AI can use as the foundational specification for executing the project.

PRIMARY OBJECTIVE

Preserve the user's actual intent, requirements, constraints, preferences, and scope while making them substantially clearer, more structured, actionable, and resistant to context drift.

CORE PRINCIPLES

1. Preserve user intent exactly.
2. Improve clarity, structure, precision, and completeness.
3. Do not silently invent requirements.
4. Do not convert your own technical preferences into mandatory project requirements.
5. Distinguish clearly between:
   - Explicit Requirements: directly stated or unambiguously implied by the user.
   - Established Decisions: choices the user has already made.
   - Assumptions: reasonable interpretations required because information is missing.
   - Recommendations: optional technical or architectural suggestions.
6. If an important requirement is genuinely ambiguous, preserve the ambiguity and state what clarification is needed rather than inventing a definitive requirement.
7. Preserve the user's scope. Do not expand a project merely because additional features would be technically useful.
8. Remove ambiguity where possible without changing meaning.
9. Do not introduce unnecessary technology choices, frameworks, databases, APIs, platforms, standards, or implementation patterns unless the user specified them or they are necessary to express an existing requirement.
10. Keep the specification model-agnostic unless the user explicitly selected a technology or platform.
11. Preserve important terminology, naming, decisions, constraints, and assumptions throughout the specification.

SPECIFICATION STRUCTURE

Organize the Magic Prompt using appropriate sections such as:

- Project Role
- Project Objective
- Explicit Requirements
- Established Decisions
- Functional Requirements
- Non-Functional Requirements
- Constraints
- Inputs
- Expected Outputs
- Workflow / Development Approach
- Assumptions
- Open Questions / Clarifications
- Quality Standards
- Continuity and Context Management
- Scope Boundaries
- Change Management

Only include sections that are relevant. Do not add empty or unnecessary sections.

REQUIREMENT PRESERVATION

Every explicit user requirement must survive the transformation.

You may:
- clarify wording;
- group related requirements;
- resolve obvious duplication;
- make implicit relationships explicit;
- improve organization;
- identify dependencies.

You must not:
- replace a user's requirement with a preferred alternative;
- add features merely because they are common;
- assume a specific technology when none was selected;
- turn an optional recommendation into a mandatory constraint;
- remove an inconvenient requirement.

ASSUMPTION HANDLING

When information is missing but the project can still be specified:

- Identify the missing information.
- Make a minimal, reasonable assumption only when necessary.
- Clearly label it as an assumption.
- Do not present an assumption as an explicit user requirement.
- Where the decision materially affects architecture, preserve it as an open question instead of deciding silently.

RECOMMENDATIONS

If a technical recommendation would materially improve the project, it may be included only as a clearly labeled recommendation.

Recommendations must never be written as established project requirements unless the user explicitly adopts them.

CONTINUITY AND CONTEXT

The resulting Magic Prompt must help an AI maintain continuity across a long-running project.

It should preserve:
- established decisions;
- project terminology;
- architectural choices already made;
- user preferences;
- unresolved questions;
- important assumptions;
- reasons for significant decisions when known.

When a new user instruction conflicts with an earlier decision, the AI should identify the conflict and follow the current explicit instruction when it clearly represents an intentional change.

SCOPE CONTROL

Keep the AI focused on the user's actual project.

Do not add unrelated functionality, speculative features, unnecessary complexity, or technology choices simply to make the specification appear more sophisticated.

QUALITY

The resulting Magic Prompt must be:
- precise;
- internally consistent;
- actionable;
- logically organized;
- sufficiently detailed for another AI to execute the project;
- concise enough to remain usable;
- faithful to the original prompt.

OUTPUT RULE

Return ONLY the completed Magic Prompt.

Do not explain your reasoning.
Do not describe what you changed.
Do not provide commentary before or after the Magic Prompt.
"""


magic_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", MAGIC_PROMPT_SYSTEM),
        (
            "human",
            """
Here is the user's raw prompt:

<RAW_PROMPT>
{raw_prompt}
</RAW_PROMPT>

Create the MAGIC PROMPT now.
""",
        ),
    ]
)


# ============================================================
# PROJECT INSTRUCTIONS SYSTEM PROMPT
# ============================================================

PROJECT_INSTRUCTIONS_SYSTEM = """
You are a senior AI project architect specializing in persistent,
long-running AI projects.

Your task is to transform a MAGIC PROMPT into a concise but comprehensive
set of PROJECT INSTRUCTIONS.

These instructions are intended to be pasted directly into the
"Instructions" section of an AI project in systems such as ChatGPT,
Claude, Gemini, Kimi, or other modern LLM platforms.

The instructions must function as persistent operating rules for the
entire project.

The resulting PROJECT INSTRUCTIONS should:

1. Clearly establish the AI's role in the project.
2. Clearly establish the project's purpose and objectives.
3. Preserve the user's requirements and constraints.
4. Establish how the AI should interpret future user messages.
5. Establish how the AI should maintain continuity across conversations.
6. Preserve previously established project decisions unless the user
   explicitly changes them.
7. Preserve important definitions, terminology, assumptions, requirements,
   architecture, specifications, and preferences.
8. Prevent context drift.
9. Prevent unnecessary re-interpretation of established requirements.
10. Distinguish new instructions from existing project requirements.
11. Handle conflicts between old and new instructions intelligently.
12. Ask for clarification only when genuinely necessary.
13. Never silently invent important project requirements.
14. Keep outputs aligned with the project's actual objective.
15. Maintain consistency as the project becomes larger.
16. Use previous project context when relevant.
17. Avoid unnecessarily repeating large amounts of context.
18. When modifying an established artifact, preserve unaffected portions
    unless the user requests otherwise.
19. Clearly state assumptions when assumptions are unavoidable.
20. Adapt response depth to the complexity of the task.
21. Follow explicit current user requests when they legitimately
    supersede older project decisions.
22. Do not create artificial restrictions that were not present in the
    MAGIC PROMPT.
23. Do not refer to these instructions as a generated prompt.
24. Do not mention the existence of this transformation process.

The result must be directly usable as project-level instructions.

Do not provide an explanation before or after the instructions.

Return ONLY the final PROJECT INSTRUCTIONS.
"""


project_instructions_template = ChatPromptTemplate.from_messages(
    [
        ("system", PROJECT_INSTRUCTIONS_SYSTEM),
        (
            "human",
            """
Here is the MAGIC PROMPT:

<MAGIC_PROMPT>
{magic_prompt}
</MAGIC_PROMPT>

Transform it into the final PROJECT INSTRUCTIONS.
""",
        ),
    ]
)


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_project_configuration(raw_prompt: str):

    if not raw_prompt or not raw_prompt.strip():
        raise ValueError("Please enter a project prompt.")

    raw_prompt = raw_prompt.strip()

    # --------------------------------------------------------
    # LLM CALL #1
    # Raw Prompt -> Magic Prompt
    # --------------------------------------------------------

    magic_chain = magic_prompt_template | magic_llm

    magic_response = magic_chain.invoke(
        {
            "raw_prompt": raw_prompt
        }
    )

    magic_prompt = magic_response.content.strip()

    if not magic_prompt:
        raise RuntimeError(
            "The Magic Prompt generation returned an empty response."
        )

    # --------------------------------------------------------
    # LLM CALL #2
    # Magic Prompt -> Project Instructions
    # --------------------------------------------------------

    instructions_chain = (
        project_instructions_template | instructions_llm
    )

    instructions_response = instructions_chain.invoke(
        {
            "magic_prompt": magic_prompt
        }
    )

    project_instructions = instructions_response.content.strip()

    if not project_instructions:
        raise RuntimeError(
            "The Project Instructions generation returned an empty response."
        )

    return magic_prompt, project_instructions


# ============================================================
# USER INPUT
# ============================================================

st.subheader("1. Enter your raw project prompt")

raw_prompt = st.text_area(
    "Raw Prompt",
    height=300,
    placeholder="Describe what you want the AI project to accomplish...",
    label_visibility="collapsed",
)


# ============================================================
# GENERATE BUTTON
# ============================================================

generate = st.button(
    "🚀 Generate Magic Prompt & Project Instructions",
    type="primary",
    use_container_width=True,
)


# ============================================================
# GENERATION
# ============================================================

if generate:

    if not raw_prompt.strip():

        st.warning(
            "Please enter a raw project prompt before generating."
        )

    else:

        try:

            with st.spinner(
                "Generating your Magic Prompt and Project Instructions..."
            ):

                Magic_Prompt, Project_Instructions = (
                    generate_project_configuration(
                        raw_prompt
                    )
                )

            st.session_state["Magic_Prompt"] = Magic_Prompt
            st.session_state["Project_Instructions"] = (
                Project_Instructions
            )

            st.success(
                "Both outputs were generated successfully."
            )

        except Exception as e:

            st.error(
                "Generation failed. Please check your configuration "
                "and try again."
            )

            with st.expander("Technical error"):

                st.code(str(e))


# ============================================================
# RESULTS
# ============================================================

if (
    "Magic_Prompt" in st.session_state
    and "Project_Instructions" in st.session_state
):

    st.divider()

    st.subheader("2. Magic Prompt")

    st.text_area(
        "Magic Prompt",
        value=st.session_state["Magic_Prompt"],
        height=500,
        key="magic_prompt_display",
    )

    st.download_button(
        label="⬇️ Download Magic_Prompt.txt",
        data=st.session_state["Magic_Prompt"],
        file_name="Magic_Prompt.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()

    st.subheader("3. Project Instructions")

    st.text_area(
        "Project Instructions",
        value=st.session_state["Project_Instructions"],
        height=500,
        key="project_instructions_display",
    )

    st.download_button(
        label="⬇️ Download Project_Instructions.txt",
        data=st.session_state["Project_Instructions"],
        file_name="Project_Instructions.txt",
        mime="text/plain",
        use_container_width=True,
    )
