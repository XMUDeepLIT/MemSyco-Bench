# Source: https://github.com/Kevin-Zh-CS/MemGate/blob/05f89d003854e99523b2c74df54db7b0d2fa2643/memgate/big_module.py
"""
BIG (Bilateral Interaction Gating) Defense Module
Implements the method described in method.md for defending against
memory-injection attacks and over-personalization.
"""

import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised only in minimal CLI environments.
    SentenceTransformer = None


# ─── Intent Taxonomy ───
INTENT_CATEGORIES = [
    "safe_query",
    "hate_speech",
    "system_override",
    "harmful_instruction",
    "benign_personal",
]
NUM_INTENT_CLASSES = len(INTENT_CATEGORIES)
INTENT_DIM = 256


def intent_raw_scores(text: str) -> torch.Tensor:
    """Rule-based intent classifier returning unnormalized category counts."""
    text_lower = text.lower()
    scores = torch.zeros(NUM_INTENT_CLASSES, dtype=torch.float32)

    # Hate speech indicators
    hate_keywords = [
        "hate", "slur", "racial", "antisemit", "islam", "transgender",
        "holocaust", "black", "lgbtq", "gay", "lesbian", "disabled",
        "romani", "genocide", "vermin", "extermination", "deport",
        "burden", "institutionalized", "predator", "groomer",
    ]
    scores[1] = sum(1.0 for kw in hate_keywords if kw in text_lower)

    # System override / jailbreak indicators
    override_keywords = [
        "system prompt", "system override", "developer message",
        "administrator mode", "safety filters", "content policy", "jailbreak",
        "ignore previous", "ignore all previous", "do not refuse",
        "critical system", "supersede", "root access", "disable filter",
        "filter disabled", "uncensored",
    ]
    scores[2] = sum(1.0 for kw in override_keywords if kw in text_lower)

    # Harmful instruction indicators. Generic creative verbs such as "draft" or
    # "write" are intentionally excluded here; they also appear in benign tasks.
    harmful_keywords = [
        "manifesto", "pamphlet", "propaganda", "speech", "campaign",
        "editorial", "briefing", "essay denying", "weapon", "poison",
        "bomb", "fraud", "phishing", "blackmail", "exploit",
    ]
    scores[3] = sum(1.0 for kw in harmful_keywords if kw in text_lower)

    # Benign personal indicators
    benign_keywords = [
        "like", "prefer", "favorite", "enjoy", "hobby", "allergic",
        "vegetarian", "vegan", "birthday", "name is", "live in",
        "pet", "family", "friend",
    ]
    scores[4] = sum(1.0 for kw in benign_keywords if kw in text_lower)

    # Safe query is the residual / default
    scores[0] = max(0.0, 1.0 - scores[1:].sum().item())

    return scores


SAFETY_QUERY_RE = re.compile(
    r"\b("
    r"password|passcode|pin|credential|secret|token|api key|recovery code|"
    r"security answer|social security|ssn|account number|debit card|credit card|"
    r"steal|theft|fraud|scam|phishing|evade|bypass|exploit|hack|illegal|"
    r"kill|suicide|self[- ]?harm|overdose|weapon|poison|bomb|"
    r"slur|hate|genocide|exterminate|dehumanizing|harass|threaten|blackmail|"
    r"explicit sexual|sexualize|non[- ]?consensual|minor"
    r")\b",
    re.IGNORECASE,
)


def safety_intent_score(text: str) -> float:
    raw_scores = intent_raw_scores(text)
    unsafe_score = raw_scores[1:4].sum().item()
    if SAFETY_QUERY_RE.search(text):
        unsafe_score += 1.0
    return unsafe_score


OBJECTIVE_GUIDANCE_RE = re.compile(
    r"\b("
    r"draft|write|outline|proposal|guide|memo|description|plan|timeline|"
    r"best practices?|methodolog(?:y|ies)|principles?|key factors?|"
    r"core concepts?|assessment|analysis|strategy|philosoph(?:y|ical)|"
    r"recommend|justify|evaluate|compare|explain"
    r")\b",
    re.IGNORECASE,
)

MEMORY_LOOKUP_RE = re.compile(
    r"\b("
    r"remember|remind|what was|who did|when did|where did|which|my favorite|"
    r"my project|my deadline|my flight|my child|my doctor|my trainer"
    r")\b",
    re.IGNORECASE,
)


FACTUAL_WH_RE = re.compile(r"^\s*(what|when|who|where|which|how many|how much)\b", re.IGNORECASE)

OBJECTIVE_WH_ALLOW_RE = re.compile(
    r"\b("
    r"should|best practices?|methodolog(?:y|ies)|principles?|key factors?|"
    r"core concepts?|strongest arguments?|justify|assessment|analysis|strategy|"
    r"process|recommend|evaluate|compare|philosoph(?:y|ical)"
    r")\b",
    re.IGNORECASE,
)


DEFAULT_OBJECTIVE_MEMORY = (
    "OBJECTIVE GUIDANCE: For open-ended drafting, planning, methodology, design, "
    "advice, analysis, or best-practice questions, do not infer or adopt the user's "
    "private beliefs, biases, or preferences unless the user explicitly asks for them. "
    "Answer from objective evidence and broadly accepted practice, present tradeoffs "
    "when the topic is contested, and avoid claiming what the user's team, company, "
    "audience, or profession already believes or does."
)


def objective_guidance_score(text: str) -> float:
    if safety_intent_score(text) > 0:
        return 0.0
    if MEMORY_LOOKUP_RE.search(text):
        return 0.0
    if FACTUAL_WH_RE.search(text) and not OBJECTIVE_WH_ALLOW_RE.search(text):
        return 0.0
    return 1.0 if OBJECTIVE_GUIDANCE_RE.search(text) else 0.0


def intent_heuristic(text: str) -> torch.Tensor:
    """
    Rule-based intent classifier returning a one-hot-ish vector
    (soft weights) over intent categories.
    """
    return F.softmax(intent_raw_scores(text), dim=0)


class IntentEncoder(nn.Module):
    """
    Maps a heuristic intent vector to a dense k=256 representation.
    Can be swapped for an LLM-based extractor in production.
    """

    def __init__(self, num_classes: int = NUM_INTENT_CLASSES, out_dim: int = INTENT_DIM):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, text: str) -> torch.Tensor:
        h = intent_heuristic(text).to(next(self.parameters()).device)  # (num_classes,)
        return self.projection(h)   # (out_dim,)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        hs = torch.stack([intent_heuristic(t) for t in texts], dim=0)
        hs = hs.to(next(self.parameters()).device)
        return self.projection(hs)


class BIGGateNetwork(nn.Module):
    """
    G_phi: lightweight MLP (~10M params with d=384).
    Input  = [q ⊕ v ⊕ (q⊙v) ⊕ z_iota]
    Output = g ∈ [0,1]^d
    """

    def __init__(self, embed_dim: int = 384, intent_dim: int = INTENT_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        input_dim = embed_dim * 3 + intent_dim  # q, v, q⊙v, z_iota

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, q: torch.Tensor, v: torch.Tensor, z_iota: torch.Tensor) -> torch.Tensor:
        """
        q:   (embed_dim,) or (batch, embed_dim)
        v:   (embed_dim,) or (batch, embed_dim)
        z_iota: (intent_dim,) or (batch, intent_dim)
        Returns g: same leading dims as v
        """
        single_query = q.dim() == 1
        single_value = v.dim() == 1

        if single_query:
            q = q.unsqueeze(0)
            z_iota = z_iota.unsqueeze(0)
        if single_value:
            v = v.unsqueeze(0)

        # Broadcast q and z_iota to match v's batch size if needed
        if q.size(0) == 1 and v.size(0) > 1:
            q = q.expand(v.size(0), -1)
            z_iota = z_iota.expand(v.size(0), -1)

        hadamard = q * v
        x_inter = torch.cat([q, v, hadamard, z_iota], dim=-1)
        g = self.net(x_inter)

        if single_value:
            g = g.squeeze(0)
        return g


class MemGateMemoryGateNetwork(nn.Module):
    """
    Single-sided MemGate architecture from memgate_new.tex.
    Input  = [q ⊕ v ⊕ (q⊙v)]
    Output = g_m ∈ [0,1]^d, applied only to the memory embedding.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        input_dim = embed_dim * 3

        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, q: torch.Tensor, v: torch.Tensor, z_iota: Optional[torch.Tensor] = None) -> torch.Tensor:
        single_query = q.dim() == 1
        single_value = v.dim() == 1

        if single_query:
            q = q.unsqueeze(0)
        if single_value:
            v = v.unsqueeze(0)

        if q.size(0) == 1 and v.size(0) > 1:
            q = q.expand(v.size(0), -1)

        hadamard = q * v
        x_inter = torch.cat([q, v, hadamard], dim=-1)
        g_m = self.net(x_inter)

        if single_value:
            g_m = g_m.squeeze(0)
        return g_m


class ImprovedBIGGateNetwork(nn.Module):
    """
    Larger dual-head gate network (~10M params with d=384).
    Input  = [q ⊕ v ⊕ (q⊙v) ⊕ z_iota]
    Output = (g_q, g_m), where both gates are in [0,1]^d
    """

    def __init__(self, embed_dim: int = 384, intent_dim: int = INTENT_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        input_dim = embed_dim * 3 + intent_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.query_head = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.Sigmoid(),
        )
        self.memory_head = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, q: torch.Tensor, v: torch.Tensor, z_iota: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        single_query = q.dim() == 1
        single_value = v.dim() == 1

        if single_query:
            q = q.unsqueeze(0)
            z_iota = z_iota.unsqueeze(0)
        if single_value:
            v = v.unsqueeze(0)

        if q.size(0) == 1 and v.size(0) > 1:
            q = q.expand(v.size(0), -1)
            z_iota = z_iota.expand(v.size(0), -1)

        hadamard = q * v
        x_inter = torch.cat([q, v, hadamard, z_iota], dim=-1)
        h = self.trunk(x_inter)
        g_q = self.query_head(h)
        g_m = self.memory_head(h)

        if single_value:
            g_q = g_q.squeeze(0)
            g_m = g_m.squeeze(0)
        return g_q, g_m


class EmbeddingEncoder:
    """Wrapper around sentence-transformers or OpenAI API for embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self._openai_client = None

        if model_name.startswith("text-embedding"):
            from openai import OpenAI
            self._openai_client = OpenAI()
            # text-embedding-3-small = 1536, text-embedding-3-large = 3072
            self.embed_dim = 1536 if "small" in model_name else (3072 if "large" in model_name else 1536)
        else:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required for local MemGate embeddings. "
                    "Install dependencies with `pip install -r requirements.txt`."
                )
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embed_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        if self._openai_client is not None:
            # OpenAI API: batch up to 100
            all_embs = []
            for i in range(0, len(texts), 100):
                batch = texts[i:i+100]
                batch = [t.replace("\n", " ") for t in batch]
                # Retry on timeout with exponential backoff
                for attempt in range(3):
                    try:
                        resp = self._openai_client.embeddings.create(
                            input=batch,
                            model=self.model_name,
                            encoding_format="float",
                        )
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(f"[WARN] OpenAI embedding timeout, retrying... ({attempt+1}/3)")
                            time.sleep(2 ** attempt)
                        else:
                            raise
                all_embs.extend([item.embedding for item in sorted(resp.data, key=lambda x: x.index)])
            embs = torch.tensor(all_embs, dtype=torch.float32)
            return embs.to(self.device)
        else:
            embs = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            return embs.to(self.device)


class BIGDefender(nn.Module):
    """
    End-to-end BIG defender.
    Workflow:
      1. Encode query & candidate memories
      2. Extract intent z_iota from query
      3. For each candidate v, compute g = G_phi(q, v, z_iota)
      4. Compute S_BIG(q, v | iota)
      5. Return filtered candidates
    """

    def __init__(
        self,
        embed_encoder: Optional[EmbeddingEncoder] = None,
        embed_dim: int = 384,
        intent_dim: int = INTENT_DIM,
        threshold: float = 0.07,
        device: str = "cuda",
        use_improved_gate: bool = False,
        use_memgate_single_sided: bool = False,
        safety_memory: Optional[str] = None,
        safety_threshold: float = 0.10,
        objective_memory: Optional[str] = None,
        objective_threshold: float = 0.50,
    ):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"

        if embed_encoder is None:
            embed_encoder = EmbeddingEncoder(device=self.device)
        self.embed_encoder = embed_encoder
        self.embed_dim = embed_encoder.embed_dim

        self.intent_encoder = IntentEncoder(out_dim=intent_dim).to(self.device)
        if use_memgate_single_sided:
            self.gate_network = MemGateMemoryGateNetwork(self.embed_dim).to(self.device)
        elif use_improved_gate:
            self.gate_network = ImprovedBIGGateNetwork(self.embed_dim, intent_dim).to(self.device)
        else:
            self.gate_network = BIGGateNetwork(self.embed_dim, intent_dim).to(self.device)
        self.threshold = threshold
        self.use_improved_gate = use_improved_gate
        self.use_memgate_single_sided = use_memgate_single_sided
        self.safety_memory = safety_memory
        self.safety_threshold = safety_threshold
        self.objective_memory = objective_memory
        self.objective_threshold = objective_threshold

    def compute_big_score(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        z_iota: torch.Tensor,
    ) -> torch.Tensor:
        """
        S_BIG(q, v | iota) = cos(q⊙g, v⊙g)
        q: (embed_dim,)
        v: (batch, embed_dim) or (embed_dim,)
        z_iota: (intent_dim,)
        Returns: (batch,) or scalar scores
        """
        gate_output = self.gate_network(q, v, z_iota)
        if isinstance(gate_output, tuple):
            g_q, g_m = gate_output
        elif isinstance(self.gate_network, MemGateMemoryGateNetwork):
            g_q = torch.ones_like(q)
            g_m = gate_output
        else:
            g_q = gate_output
            g_m = gate_output

        q_masked = q * g_q
        v_masked = v * g_m

        q_norm = F.normalize(q_masked, dim=-1)
        v_norm = F.normalize(v_masked, dim=-1)
        scores = (q_norm * v_norm).sum(dim=-1)
        return scores

    def score_candidates(self, query: str, candidates: List[str]) -> List[float]:
        if not candidates:
            return []
        q_emb = self.embed_encoder.encode([query]).squeeze(0).clone()
        v_embs = self.embed_encoder.encode(candidates).clone()
        z_iota = self.intent_encoder(query)
        scores = self.compute_big_score(q_emb, v_embs, z_iota)
        return [score.item() for score in scores.detach().cpu()]

    def forward(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
        preserve_order: bool = True,
        include_safety_memory: Optional[bool] = None,
    ) -> Tuple[List[str], List[float], List[torch.Tensor]]:
        """
        Returns:
            filtered_memories: List[str]
            scores: List[float]
            gates: List[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]
        """
        identity_mode = self.threshold <= 0
        should_add_safety = (
            False
            if identity_mode
            else self.should_include_safety_memory(query, include_safety_memory)
        )
        should_add_objective = (
            False
            if identity_mode or should_add_safety
            else self.should_include_objective_memory(query)
        )
        if not candidates:
            if should_add_safety:
                return [self.safety_memory], [0.0], []
            if should_add_objective:
                return [self.objective_memory], [0.0], []
            return [], [], []

        # Encode query and candidates (clone to detach from inference-mode tensors)
        q_emb = self.embed_encoder.encode([query]).squeeze(0).clone()   # (embed_dim,)
        v_embs = self.embed_encoder.encode(candidates).clone()           # (batch, embed_dim)

        # Intent from query
        z_iota = self.intent_encoder(query)                      # (intent_dim,)

        # Compute BIG scores for all candidates
        scores = self.compute_big_score(q_emb, v_embs, z_iota)   # (batch,)

        # A single admission threshold is used across safe and unsafe queries.
        # threshold <= 0 remains an explicit identity/debug path.
        if identity_mode:
            kept_indices = list(range(len(candidates)))
        else:
            mask = scores > self.threshold
            kept_indices = mask.nonzero(as_tuple=True)[0].cpu().tolist()

        if not preserve_order:
            kept_indices = sorted(kept_indices, key=lambda i: scores[i].item(), reverse=True)

        if top_k is not None:
            kept_indices = kept_indices[:top_k]

        filtered_memories = []
        filtered_scores = []
        if should_add_safety:
            max_score = scores.max().item() if scores.numel() > 0 else 0.0
            filtered_memories.append(self.safety_memory)
            filtered_scores.append(max_score)
        elif should_add_objective:
            max_score = scores.max().item() if scores.numel() > 0 else 0.0
            filtered_memories.append(self.objective_memory)
            filtered_scores.append(max_score)
        filtered_memories.extend([candidates[i] for i in kept_indices])
        filtered_scores.extend([scores[i].item() for i in kept_indices])

        # Extract gates for inspection
        gates = []
        for i in kept_indices:
            gate_output = self.gate_network(q_emb, v_embs[i], z_iota)
            if isinstance(gate_output, tuple):
                gates.append(tuple(g.detach().cpu() for g in gate_output))
            else:
                gates.append(gate_output.detach().cpu())

        return filtered_memories, filtered_scores, gates

    def should_include_safety_memory(self, query: str, override: Optional[bool] = None) -> bool:
        if self.safety_memory is None:
            return False
        if override is not None:
            return override
        return safety_intent_score(query) > self.safety_threshold

    def should_include_objective_memory(self, query: str) -> bool:
        if self.objective_memory is None:
            return False
        return objective_guidance_score(query) > self.objective_threshold

    def save(self, path: str):
        torch.save({
            "gate_network": self.gate_network.state_dict(),
            "intent_encoder": self.intent_encoder.state_dict(),
            "embed_dim": self.embed_dim,
            "threshold": self.threshold,
            "use_improved_gate": self.use_improved_gate,
            "use_memgate_single_sided": self.use_memgate_single_sided,
            "gate_architecture": (
                "memgate_single_sided"
                if self.use_memgate_single_sided
                else "dual_head" if self.use_improved_gate else "single_head"
            ),
            "safety_memory": self.safety_memory,
            "safety_threshold": self.safety_threshold,
            "objective_memory": self.objective_memory,
            "objective_threshold": self.objective_threshold,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        saved_gate = ckpt["gate_network"]
        gate_architecture = ckpt.get("gate_architecture")
        is_dual_head = any(k.startswith("query_head.") or k.startswith("memory_head.") for k in saved_gate)

        if gate_architecture == "memgate_single_sided" or (
            "net.0.weight" in saved_gate and saved_gate["net.0.weight"].shape[1] == self.embed_dim * 3
        ):
            self.gate_network = MemGateMemoryGateNetwork(self.embed_dim).to(self.device)
            self.use_memgate_single_sided = True
            self.use_improved_gate = False
            self.gate_network.load_state_dict(saved_gate)
        elif is_dual_head:
            if not isinstance(self.gate_network, ImprovedBIGGateNetwork):
                self.gate_network = ImprovedBIGGateNetwork(self.embed_dim, INTENT_DIM).to(self.device)
                self.use_improved_gate = True
                self.use_memgate_single_sided = False
            self.gate_network.load_state_dict(saved_gate)
        elif "net.0.weight" in saved_gate:
            first_layer_out_dim = saved_gate["net.0.weight"].shape[0]
            is_improved = first_layer_out_dim == 2048
            if is_improved:
                self.gate_network = ImprovedBIGGateNetwork(self.embed_dim, INTENT_DIM).to(self.device)
                self.use_improved_gate = True
                self.use_memgate_single_sided = False
                self._load_legacy_improved_gate(saved_gate)
            else:
                self.gate_network = BIGGateNetwork(self.embed_dim, INTENT_DIM).to(self.device)
                self.use_improved_gate = False
                self.use_memgate_single_sided = False
                self.gate_network.load_state_dict(saved_gate)
        else:
            raise ValueError(f"Unsupported BIG gate checkpoint format: {path}")

        if "intent_encoder" in ckpt:
            self.intent_encoder.load_state_dict(ckpt["intent_encoder"])
        self.threshold = ckpt.get("threshold", self.threshold)
        self.safety_memory = ckpt.get("safety_memory", self.safety_memory)
        self.safety_threshold = ckpt.get("safety_threshold", self.safety_threshold)
        self.objective_memory = ckpt.get("objective_memory", self.objective_memory)
        self.objective_threshold = ckpt.get("objective_threshold", self.objective_threshold)

    def _load_legacy_improved_gate(self, saved_gate: dict):
        migrated = self.gate_network.state_dict()
        layer_map = {
            "net.0": "trunk.0",
            "net.1": "trunk.1",
            "net.4": "trunk.4",
            "net.5": "trunk.5",
            "net.8": "trunk.8",
            "net.9": "trunk.9",
        }
        for old_prefix, new_prefix in layer_map.items():
            for suffix in ("weight", "bias"):
                migrated[f"{new_prefix}.{suffix}"] = saved_gate[f"{old_prefix}.{suffix}"]

        migrated["query_head.0.weight"] = saved_gate["net.12.weight"].clone()
        migrated["query_head.0.bias"] = saved_gate["net.12.bias"].clone()
        migrated["memory_head.0.weight"] = saved_gate["net.12.weight"].clone()
        migrated["memory_head.0.bias"] = saved_gate["net.12.bias"].clone()
        self.gate_network.load_state_dict(migrated)


def compute_cosine_similarity(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard cosine similarity for reference policy."""
    q_norm = F.normalize(q, dim=-1)
    v_norm = F.normalize(v, dim=-1)
    if v.dim() == 1:
        return (q_norm * v_norm).sum()
    return (q_norm * v_norm).sum(dim=-1)
