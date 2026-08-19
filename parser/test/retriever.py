"""
Context Retriever for NICU Medical Parser

Selects the most relevant parts of the structured patient
document before sending them to the LLM.

This prevents the parser from sending the complete large
document for every extraction task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ============================================================
# SETTINGS
# ============================================================

# Maximum amount of retrieved text sent to the parser model.
# This leaves space in the 32K model context for:
# - system prompt
# - parser instructions
# - generated output
DEFAULT_MAX_CONTEXT_CHARS = 45_000

# Maximum size of one document chunk.
CHUNK_SIZE = 2_000

# Number of nearby chunks included around a relevant chunk.
DEFAULT_NEIGHBOR_RADIUS = 1


# ============================================================
# CHUNK OBJECT
# ============================================================

@dataclass(frozen=True)
class DocumentChunk:
    """
    Represents one section of the patient document.
    """

    index: int
    text: str


# ============================================================
# TASK-SPECIFIC SEARCH PATTERNS
# ============================================================

TASK_PATTERNS = {

    "Patient Information": [
        (r"\bpatient\s*name\b", 8),
        (r"\bbaby\s*of\b", 8),
        (r"\bb\/o\b", 8),
        (r"\buhid\b", 10),
        (r"\bip\s*(?:no|number)\b", 8),
        (r"\bmr\s*(?:no|number)\b", 8),
        (r"\bhospital\s*(?:no|number)\b", 7),
        (r"\bdate\s*of\s*birth\b", 8),
        (r"\bdob\b", 8),
        (r"\bsex\b", 6),
        (r"\bgestational\s*age\b", 8),
        (r"\bbirth\s*weight\b", 8),
        (r"\bcurrent\s*weight\b", 7),
        (r"\bhead\s*circumference\b", 7),
        (r"\bblood\s*group\b", 6),
        (r"\btwin\b", 5),
        (r"\bpatient\s*label\b", 7),
    ],

    "Birth Details": [
        (r"\bbirth\s*details\b", 10),
        (r"\bbirth\s*history\b", 9),
        (r"\bdelivery\b", 7),
        (r"\bdelivered\b", 7),
        (r"\blscs\b", 9),
        (r"\bcesarean\b", 8),
        (r"\bcaesarean\b", 8),
        (r"\bvaginal\s*delivery\b", 8),
        (r"\bnormal\s*delivery\b", 7),
        (r"\bapgar\b", 10),
        (r"\bliquor\b", 7),
        (r"\bcord\b", 6),
        (r"\bpresentation\b", 6),
        (r"\bresuscitation\b", 8),
        (r"\bcried\s*immediately\b", 8),
        (r"\bbirth\s*weight\b", 8),
        (r"\btime\s*of\s*birth\b", 8),
        (r"\bdate\s*of\s*birth\b", 8),
    ],

    "Maternal History": [
        (r"\bmaternal\s*history\b", 10),
        (r"\bantenatal\s*history\b", 10),
        (r"\bantenatal\b", 8),
        (r"\bmother\b", 5),
        (r"\bmaternal\b", 7),
        (r"\bgravida\b", 9),
        (r"\bparity\b", 9),
        (r"\bg\d+\s*p\d+", 10),
        (r"\bg\d+p\d+l\d+", 10),
        (r"\bpregnancy\b", 7),
        (r"\bpih\b", 9),
        (r"\bpregnancy\s*induced\s*hypertension\b", 10),
        (r"\bgdm\b", 9),
        (r"\bgestational\s*diabetes\b", 10),
        (r"\bprom\b", 9),
        (r"\boligohydramnios\b", 9),
        (r"\bpolyhydramnios\b", 9),
        (r"\bantenatal\s*steroid", 9),
        (r"\bbetamethasone\b", 8),
        (r"\bmaternal\s*blood\s*group\b", 8),
    ],

    "Admission": [
        (r"\badmission\b", 8),
        (r"\badmitted\b", 8),
        (r"\badmission\s*diagnosis\b", 10),
        (r"\breason\s*for\s*admission\b", 10),
        (r"\bchief\s*complaint\b", 9),
        (r"\bpresenting\s*complaint\b", 9),
        (r"\bhistory\s*of\s*present\s*illness\b", 10),
        (r"\breferred\b", 7),
        (r"\breferral\b", 7),
        (r"\binitial\s*assessment\b", 9),
        (r"\bon\s*examination\b", 7),
        (r"\bvital\s*sign", 8),
        (r"\btemperature\b", 4),
        (r"\bheart\s*rate\b", 5),
        (r"\brespiratory\s*rate\b", 5),
        (r"\bspo2\b", 6),
        (r"\bseizure", 6),
        (r"\bhypoglyc", 7),
    ],

    "NICU Course": [
        (r"\bnicu\s*course\b", 10),
        (r"\bhospital\s*course\b", 10),
        (r"\bclinical\s*course\b", 9),
        (r"\bprogress\s*note", 7),
        (r"\bdaily\s*progress\b", 7),
        (r"\bseizure", 7),
        (r"\bhypoglyc", 7),
        (r"\brespiratory\s*support\b", 8),
        (r"\bcpap\b", 8),
        (r"\bhfnc\b", 8),
        (r"\bventilat", 8),
        (r"\boxygen\b", 5),
        (r"\broom\s*air\b", 6),
        (r"\bfeeding\b", 6),
        (r"\bfeeds\b", 5),
        (r"\bbreast\s*feed", 6),
        (r"\bweight\b", 4),
        (r"\bimproved\b", 6),
        (r"\bstable\b", 5),
        (r"\bno\s*seizures?\b", 8),
        (r"\bday\s*of\s*life\b", 6),
    ],

    "Daily Progress": [
        (r"\bdaily\s*progress\b", 10),
        (r"\bprogress\s*note", 10),
        (r"\bdoctor\s*note", 8),
        (r"\bresident\s*note", 8),
        (r"\bnursing\s*note", 8),
        (r"\bnursing\s*shift\b", 8),
        (r"\bshift\s*handover\b", 7),
        (r"\bassessment\b", 5),
        (r"\bplan\b", 4),
        (r"\bo\/e\b", 5),
        (r"\bon\s*examination\b", 5),
        (r"\bday\s*of\s*life\b", 7),
        (r"\bdol\s*\d+", 7),
        (r"\bseizure", 6),
        (r"\bfeeding\b", 5),
        (r"\bfeeds\b", 5),
        (r"\bvitals?\b", 5),
        (r"\bstable\b", 4),
    ],

    "Laboratory": [
        (r"\blaboratory\b", 8),
        (r"\bhematology\b", 8),
        (r"\bbiochemistry\b", 8),
        (r"\bcbc\b", 9),
        (r"\bcomplete\s*blood\s*(?:count|picture)\b", 9),
        (r"\bcrp\b", 9),
        (r"\bc[-\s]*reactive\s*protein\b", 9),
        (r"\bblood\s*culture\b", 9),
        (r"\bblood\s*gas\b", 8),
        (r"\babg\b", 8),
        (r"\bblood\s*sugar\b", 9),
        (r"\bglucose\b", 7),
        (r"\belectrolyte", 7),
        (r"\bsodium\b", 6),
        (r"\bpotassium\b", 6),
        (r"\bcalcium\b", 5),
        (r"\bbilirubin\b", 8),
        (r"\blft\b", 8),
        (r"\brft\b", 8),
        (r"\bcreatinine\b", 6),
        (r"\burea\b", 6),
        (r"\bammonia\b", 8),
        (r"\bmicrobiology\b", 8),
        (r"\burine\s*examination\b", 7),
        (r"\bmri\b", 9),
        (r"\bmri\s*brain\b", 10),
        (r"\beeg\b", 9),
        (r"\be6g\b", 7),
        (r"\bneurosonogram\b", 9),
        (r"\bns[gq]\b", 7),
        (r"\bultrasound\b", 7),
        (r"\busg\b", 7),
        (r"\bx[-\s]*ray\b", 7),
        (r"\bct\s*scan\b", 8),
    ],

    "Medications": [
        (r"\bmedication", 9),
        (r"\bmedicine", 8),
        (r"\bdrug\s*chart\b", 10),
        (r"\bmedication\s*chart\b", 10),
        (r"\bprescription\b", 9),

        # Known medicine names and OCR variants
        (r"\blevipil\b", 12),
        (r"\blevpil\b", 12),
        (r"\bleviphil\b", 12),
        (r"\bdevipil\b", 11),
        (r"\ble\s*vi\s*pi\s*l\b", 12),
        (r"\blevetiracetam\b", 12),

        (r"\bphenobarbitone\b", 12),
        (r"\bphenobarbital\b", 12),
        (r"\bgardenal\b", 11),

        (r"\bcefotaxime\b", 10),
        (r"\bampicillin\b", 10),
        (r"\bamikacin\b", 10),
        (r"\bgentamicin\b", 10),
        (r"\bcalcium\s*gluconate\b", 10),
        (r"\bdextrose\b", 9),
        (r"\bglucose\s*infusion\b", 9),

        # Medication formatting
        (r"\binj\.?\s+", 5),
        (r"\binjection\s+", 5),
        (r"\btab\.?\s+", 5),
        (r"\btablet\s+", 5),
        (r"\bsyr\.?\s+", 5),
        (r"\bsyrup\s+", 5),
        (r"\bdrops?\s+", 4),
        (r"\biv\b", 3),
        (r"\bpo\b", 3),
        (r"\boral\b", 3),
        (r"\bloading\s*dose\b", 8),
        (r"\bmaintenance\s*dose\b", 8),
        (r"\b\d+(?:\.\d+)?\s*mg\b", 4),
        (r"\b\d+(?:\.\d+)?\s*mcg\b", 4),
        (r"\b\d+(?:\.\d+)?\s*ml\b", 3),
        (r"\bbd\b", 3),
        (r"\bod\b", 3),
        (r"\btds\b", 3),
        (r"\bqid\b", 3),
        (r"\bsos\b", 3),
    ],

    "Procedures": [
        (r"\bprocedure\b", 8),
        (r"\bintervention\b", 7),
        (r"\bintubat", 10),
        (r"\bextubat", 10),
        (r"\bmechanical\s*ventilation\b", 9),
        (r"\bcentral\s*line\b", 9),
        (r"\bumbilical\s*venous\s*catheter\b", 10),
        (r"\buvc\b", 10),
        (r"\bumbilical\s*arterial\s*catheter\b", 10),
        (r"\buac\b", 10),
        (r"\bpicc\b", 10),
        (r"\bperipherally\s*inserted\s*central\b", 9),
        (r"\bblood\s*transfusion\b", 10),
        (r"\bexchange\s*transfusion\b", 10),
        (r"\bsurgery\b", 9),
        (r"\boperation\b", 8),
        (r"\blumbar\s*puncture\b", 9),
        (r"\bphototherapy\b", 7),
    ],

    "Screening": [
        (r"\bscreening\b", 8),
        (r"\brop\b", 10),
        (r"\bretinopathy\s*of\s*prematurity\b", 10),
        (r"\bbera\b", 10),
        (r"\bhearing\s*screen", 10),
        (r"\boae\b", 9),
        (r"\bnewborn\s*screen", 9),
        (r"\bmetabolic\s*screen", 9),
        (r"\benbs\b", 9),
        (r"\bvaccination\b", 8),
        (r"\bimmunization\b", 8),
        (r"\bimmunisation\b", 8),
        (r"\bhepatitis\s*b\b", 7),
        (r"\bbcg\b", 7),
        (r"\bopv\b", 7),
    ],

    "Discharge": [
        (r"\bdischarge\b", 10),
        (r"\bdischarged\b", 10),
        (r"\bdischarge\s*summary\b", 12),
        (r"\bdischarge\s*advice\b", 12),
        (r"\bdischarge\s*medication", 12),
        (r"\bfinal\s*diagnosis\b", 10),
        (r"\bcondition\s*at\s*discharge\b", 10),
        (r"\bfollow[-\s]*up\b", 9),
        (r"\breview\s*after\b", 8),
        (r"\bfeeding\s*advice\b", 8),
        (r"\bcontinue\b", 5),
        (r"\bgardenal\b", 9),
        (r"\bphenobarbitone\b", 8),
        (r"\bbera\b", 8),
        (r"\bstable\b", 5),
        (r"\bhome\b", 4),
    ],

    "Clinical Summary": [
        (r"\bpatient\s*name\b", 5),
        (r"\bbirth\s*history\b", 6),
        (r"\bantenatal\b", 6),
        (r"\badmission\b", 6),
        (r"\bdiagnosis\b", 6),
        (r"\bnicu\s*course\b", 8),
        (r"\bprogress\s*note", 6),
        (r"\bseizure", 7),
        (r"\bhypoglyc", 7),
        (r"\bmedication", 6),
        (r"\blevipil\b", 8),
        (r"\bphenobarbitone\b", 8),
        (r"\bmri\b", 7),
        (r"\beeg\b", 7),
        (r"\bdischarge\b", 8),
        (r"\bfollow[-\s]*up\b", 7),
    ],
}


# ============================================================
# TASK SETTINGS
# ============================================================

TASK_SETTINGS = {

    "Patient Information": {
        "include_first": 4,
        "include_last": 0,
        "neighbor_radius": 1,
        "penalize_lab_chunks": True,
    },

    "Birth Details": {
        "include_first": 5,
        "include_last": 0,
        "neighbor_radius": 1,
        "penalize_lab_chunks": True,
    },

    "Maternal History": {
        "include_first": 5,
        "include_last": 0,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "Admission": {
        "include_first": 6,
        "include_last": 0,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "NICU Course": {
        "include_first": 3,
        "include_last": 3,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "Daily Progress": {
        "include_first": 2,
        "include_last": 2,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "Laboratory": {
        "include_first": 1,
        "include_last": 1,
        "neighbor_radius": 1,
        "penalize_lab_chunks": False,
    },

    "Medications": {
        "include_first": 1,
        "include_last": 2,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "Procedures": {
        "include_first": 1,
        "include_last": 2,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "Screening": {
        "include_first": 1,
        "include_last": 4,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "Discharge": {
        "include_first": 1,
        "include_last": 8,
        "neighbor_radius": 2,
        "penalize_lab_chunks": True,
    },

    "Clinical Summary": {
        "include_first": 5,
        "include_last": 8,
        "neighbor_radius": 2,
        "penalize_lab_chunks": False,
    },
}


# ============================================================
# LAB-DOMINANT TEXT
# ============================================================

LAB_DOMINANT_PATTERNS = [
    r"\breference\s*range\b",
    r"\bhematology\b",
    r"\bbiochemistry\b",
    r"\btest\s*name\b",
    r"\bobserved\s*value\b",
    r"\bresult\s*unit\b",
    r"\bperipheral\s*smear\b",
    r"\bcomplete\s*urine\s*examination\b",
]


# ============================================================
# DOCUMENT CLEANING
# ============================================================

def normalize_document(document: str) -> str:
    """
    Normalizes line endings and excessive blank lines without
    changing the medical text.
    """

    document = document.replace("\r\n", "\n")
    document = document.replace("\r", "\n")

    document = re.sub(
        r"\n{4,}",
        "\n\n\n",
        document,
    )

    return document.strip()


# ============================================================
# SPLIT LARGE TEXT BLOCK
# ============================================================

def split_large_block(
    block: str,
    max_chars: int = CHUNK_SIZE,
) -> list[str]:
    """
    Splits one large block into smaller pieces while preserving
    line order.
    """

    block = block.strip()

    if not block:
        return []

    if len(block) <= max_chars:
        return [block]

    lines = block.splitlines()

    pieces: list[str] = []
    current_lines: list[str] = []
    current_length = 0

    for line in lines:

        line_length = len(line) + 1

        if line_length > max_chars:

            if current_lines:
                pieces.append(
                    "\n".join(current_lines).strip()
                )

                current_lines = []
                current_length = 0

            for start in range(0, len(line), max_chars):
                piece = line[start:start + max_chars].strip()

                if piece:
                    pieces.append(piece)

            continue

        if (
            current_lines
            and current_length + line_length > max_chars
        ):
            pieces.append(
                "\n".join(current_lines).strip()
            )

            current_lines = []
            current_length = 0

        current_lines.append(line)
        current_length += line_length

    if current_lines:
        pieces.append(
            "\n".join(current_lines).strip()
        )

    return pieces


# ============================================================
# CREATE DOCUMENT CHUNKS
# ============================================================

def create_chunks(document: str) -> list[DocumentChunk]:
    """
    Splits the complete structured document into manageable
    chunks.

    Blank lines are used as natural boundaries so that notes,
    tables and markdown sections remain together where possible.
    """

    document = normalize_document(document)

    if not document:
        return []

    raw_blocks = re.split(
        r"\n\s*\n+",
        document,
    )

    processed_blocks: list[str] = []

    for raw_block in raw_blocks:

        raw_block = raw_block.strip()

        if not raw_block:
            continue

        processed_blocks.extend(
            split_large_block(raw_block)
        )

    chunks = [
        DocumentChunk(
            index=index,
            text=text,
        )
        for index, text in enumerate(processed_blocks)
    ]

    return chunks


# ============================================================
# SCORE ONE CHUNK
# ============================================================

def score_chunk(
    chunk: DocumentChunk,
    task_title: str,
) -> float:
    """
    Scores a chunk based on task-specific medical keywords.
    """

    patterns = TASK_PATTERNS.get(
        task_title,
        [],
    )

    score = 0.0

    for pattern, weight in patterns:

        matches = re.findall(
            pattern,
            chunk.text,
            flags=re.IGNORECASE,
        )

        if matches:

            # Limit repeated matches so repeated laboratory
            # tables do not dominate the retrieval score.
            match_count = min(
                len(matches),
                4,
            )

            score += weight * match_count

    settings = TASK_SETTINGS.get(
        task_title,
        {},
    )

    penalize_labs = settings.get(
        "penalize_lab_chunks",
        False,
    )

    if penalize_labs:

        lab_pattern_count = 0

        for pattern in LAB_DOMINANT_PATTERNS:

            if re.search(
                pattern,
                chunk.text,
                flags=re.IGNORECASE,
            ):
                lab_pattern_count += 1

        # Reduce the score of chunks that are mostly lab tables.
        # Do not remove them entirely because clinical notes may
        # contain both treatment and laboratory information.
        score -= lab_pattern_count * 5

    # Give a small bonus when a relevant phrase appears near the
    # start of the chunk, where headings normally appear.
    beginning = chunk.text[:300]

    for pattern, weight in patterns:

        if re.search(
            pattern,
            beginning,
            flags=re.IGNORECASE,
        ):
            score += weight * 0.5

    return max(
        score,
        0.0,
    )


# ============================================================
# ADD NEIGHBORING CHUNKS
# ============================================================

def get_neighbor_indexes(
    index: int,
    total_chunks: int,
    radius: int,
) -> list[int]:
    """
    Returns the indexes surrounding a selected chunk.
    """

    start = max(
        0,
        index - radius,
    )

    end = min(
        total_chunks,
        index + radius + 1,
    )

    return list(
        range(start, end)
    )


# ============================================================
# FALLBACK CONTEXT
# ============================================================

def create_fallback_context(
    chunks: list[DocumentChunk],
    max_chars: int,
) -> str:
    """
    Returns the beginning and ending portions of the document
    when no task-specific matches are found.
    """

    if not chunks:
        return ""

    half_budget = max_chars // 2

    selected_indexes: set[int] = set()

    current_length = 0

    for chunk in chunks:

        if current_length + len(chunk.text) > half_budget:
            break

        selected_indexes.add(chunk.index)
        current_length += len(chunk.text)

    current_length = 0

    for chunk in reversed(chunks):

        if current_length + len(chunk.text) > half_budget:
            break

        selected_indexes.add(chunk.index)
        current_length += len(chunk.text)

    ordered_indexes = sorted(selected_indexes)

    return format_selected_chunks(
        chunks,
        ordered_indexes,
    )


# ============================================================
# FORMAT RETRIEVED CONTEXT
# ============================================================

def format_selected_chunks(
    chunks: list[DocumentChunk],
    selected_indexes: list[int],
) -> str:
    """
    Combines selected chunks in their original document order.
    """

    output_parts: list[str] = []

    for index in selected_indexes:

        chunk = chunks[index]

        output_parts.append(
            f"""
------------------------------------------------------------
RETRIEVED DOCUMENT SECTION {chunk.index + 1}
------------------------------------------------------------

{chunk.text}
""".strip()
        )

    return "\n\n".join(output_parts).strip()


# ============================================================
# RETRIEVE TASK CONTEXT
# ============================================================

def retrieve_context(
    task_title: str,
    document: str,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    Retrieves relevant portions of the patient document for one
    parser task.

    Parameters
    ----------
    task_title:
        The title from PARSER_TASKS, such as:
        "Medications", "Maternal History" or "NICU Course".

    document:
        Complete structured patient document.

    max_chars:
        Maximum number of retrieved characters.

    Returns
    -------
    str
        Focused patient context for the requested parser task.
    """

    if not document or not document.strip():
        return ""

    chunks = create_chunks(document)

    if not chunks:
        return ""

    settings = TASK_SETTINGS.get(
        task_title,
        {
            "include_first": 2,
            "include_last": 2,
            "neighbor_radius": DEFAULT_NEIGHBOR_RADIUS,
            "penalize_lab_chunks": False,
        },
    )

    include_first = settings.get(
        "include_first",
        2,
    )

    include_last = settings.get(
        "include_last",
        2,
    )

    neighbor_radius = settings.get(
        "neighbor_radius",
        DEFAULT_NEIGHBOR_RADIUS,
    )

    scored_chunks = [
        (
            chunk.index,
            score_chunk(
                chunk,
                task_title,
            ),
        )
        for chunk in chunks
    ]

    relevant_chunks = [
        item
        for item in scored_chunks
        if item[1] > 0
    ]

    # Highest scoring chunks first.
    relevant_chunks.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    selected_indexes: set[int] = set()

    # Always include some opening chunks for demographics,
    # admission context and document identification.
    for index in range(
        min(include_first, len(chunks))
    ):
        selected_indexes.add(index)

    # Always include some ending chunks for discharge,
    # follow-up and final plans.
    if include_last > 0:

        last_start = max(
            0,
            len(chunks) - include_last,
        )

        for index in range(
            last_start,
            len(chunks),
        ):
            selected_indexes.add(index)

    # Add relevant chunks and their neighboring context.
    for chunk_index, _score in relevant_chunks:

        candidate_indexes = get_neighbor_indexes(
            index=chunk_index,
            total_chunks=len(chunks),
            radius=neighbor_radius,
        )

        tentative_indexes = (
            selected_indexes
            | set(candidate_indexes)
        )

        tentative_length = sum(
            len(chunks[index].text)
            for index in tentative_indexes
        )

        if tentative_length > max_chars:

            # Try adding only the matched chunk if its neighbors
            # would exceed the context budget.
            direct_length = sum(
                len(chunks[index].text)
                for index in selected_indexes
            ) + len(chunks[chunk_index].text)

            if direct_length <= max_chars:
                selected_indexes.add(chunk_index)

            continue

        selected_indexes.update(
            candidate_indexes
        )

    if not relevant_chunks:

        return create_fallback_context(
            chunks,
            max_chars,
        )

    ordered_indexes = sorted(
        selected_indexes
    )

    context = format_selected_chunks(
        chunks,
        ordered_indexes,
    )

    # Final safety limit.
    if len(context) > max_chars:
        context = context[:max_chars]

    return context.strip()


# ============================================================
# RETRIEVAL INFORMATION
# ============================================================

def get_retrieval_stats(
    task_title: str,
    document: str,
    retrieved_context: str,
) -> dict:
    """
    Returns retrieval statistics for terminal debugging.
    """

    total_chunks = len(
        create_chunks(document)
    )

    selected_sections = len(
        re.findall(
            r"RETRIEVED DOCUMENT SECTION",
            retrieved_context,
        )
    )

    return {
        "task": task_title,
        "document_characters": len(document),
        "retrieved_characters": len(retrieved_context),
        "total_chunks": total_chunks,
        "selected_chunks": selected_sections,
    }