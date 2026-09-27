"""Local recorded-fixture contract, deliberately NOT a second Foundation ABI."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

SCHEMA = "p6-t01-recorded.draft.v1"
POLICY = {
    "scope": "recorded-reference-only",
    "foundation_compatibility": "UNVERIFIED",
    "return_payload": "already-weighted-per-slot-rows",
    "route_weight_applied_count": 1,
    "topk": 6,
    "slot_order": [0, 1, 2, 3, 4, 5],
    "invalid": "skip; all-invalid accumulator is positive-zero",
    "accumulator_initialization": "first-valid-slot-copy",
    "accumulator_dtype": "fp32",
    "forward_input_dtype": "bf16",
    "forward_merge": ["routed", "shared-once", "residual-once"],
    "forward_output_dtype": "bf16-rne",
    "backward_input_dtype": "fp32",
    "backward_output_dtype": "fp32",
    "backward_shared_boundary": "must-equal-expert-input-boundary",
    "residual_gradient": "external-P1-fork-join; never-added-to-expert-input-dx",
    "non_finite": "reject-in-active-inputs-and-results",
    "subnormal": "unsupported-in-this-draft-profile",
    "production_provider": "UNSUPPORTED_CAPABILITY",
    "unfrozen_items": [
        "Foundation concrete ABI/version adapter and owner approval",
        "P1 residual and P3 gate-gradient wiring",
        "P4/P5 backward payload dtype and all cast points",
        "invalid-slot, signed-zero and first-valid semantics owner approval",
    ],
}


class ContractError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(ok, code, detail):
    if not ok:
        raise ContractError(code, detail)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys), "SCHEMA_MISMATCH", label)


def nonempty(value, label):
    require(type(value) is str and bool(value), "IDENTITY_DRIFT", label)


@dataclass(frozen=True)
class Context:
    run_id: str
    microbatch_id: str
    forward_id: str
    checkpoint_id: str

    def validate(self):
        for key, value in asdict(self).items():
            nonempty(value, key)


@dataclass(frozen=True)
class Plan:
    schema: str
    context: Context
    # Opaque references supplied by the producer; this kit does not redefine it.
    route_ref: tuple
    exchange_ref: tuple
    token_ids: tuple
    valid_slots: tuple
    # Each physical row holds (global_token_id, topk_slot, valid).
    inverse_map: tuple
    hidden_size: int
    gradient_boundary: str
    policy_hash: str

    def validate(self, expected_context):
        require(self.schema == SCHEMA, "SCHEMA_MISMATCH", self.schema)
        require(type(self.context) is Context, "SCHEMA_MISMATCH", "context")
        self.context.validate()
        expected_context.validate()
        require(self.context == expected_context, "STALE_RUN_METADATA", "context mismatch")
        require(self.policy_hash == digest(POLICY), "SCHEMA_MISMATCH", "arithmetic policy")
        for ref in (self.route_ref, self.exchange_ref):
            require(type(ref) is tuple and len(ref) == 2, "SCHEMA_MISMATCH", "upstream ref")
            nonempty(ref[0], "upstream schema")
            require(type(ref[1]) is str and re.fullmatch(r"[0-9a-f]{64}", ref[1]) is not None,
                    "SCHEMA_MISMATCH", "upstream fingerprint")
        require(type(self.hidden_size) is int and self.hidden_size > 0,
                "UNSUPPORTED_GEOMETRY", "hidden_size")
        nonempty(self.gradient_boundary, "gradient boundary")
        require(type(self.token_ids) is tuple and all(type(t) is int and t >= 0 for t in self.token_ids),
                "INVALID_DISCRETE_PLAN", "global token IDs")
        require(len(set(self.token_ids)) == len(self.token_ids), "INVALID_DISCRETE_PLAN", "duplicate token")
        require(type(self.valid_slots) is tuple and len(self.valid_slots) == len(self.token_ids),
                "INVALID_DISCRETE_PLAN", "valid mask rows")
        for mask in self.valid_slots:
            require(type(mask) is tuple and len(mask) == 6 and all(type(v) is bool for v in mask),
                    "INVALID_DISCRETE_PLAN", "valid mask must contain six bools")
        expected = {(t, s) for t, m in zip(self.token_ids, self.valid_slots)
                    for s in range(6) if m[s]}
        require(type(self.inverse_map) is tuple, "INVALID_DISCRETE_PLAN", "inverse_map type")
        seen = set()
        for row in self.inverse_map:
            require(type(row) is tuple and len(row) == 3, "INVALID_DISCRETE_PLAN", "inverse row")
            t, s, valid = row
            require(type(t) is int and type(s) is int and type(valid) is bool,
                    "INVALID_DISCRETE_PLAN", "inverse field types")
            if not valid:
                require((t, s) == (-1, -1), "INVALID_DISCRETE_PLAN", "padding sentinel")
                continue
            key = (t, s)
            require(key in expected and key not in seen, "INVALID_DISCRETE_PLAN", f"unexpected/duplicate {key}")
            seen.add(key)
        require(seen == expected, "INVALID_DISCRETE_PLAN", "missing valid slot")

    def to_dict(self):
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        exact_keys(value, cls.__dataclass_fields__, "Plan fields")
        exact_keys(value["context"], Context.__dataclass_fields__, "Context fields")
        try:
            return cls(
                schema=value["schema"], context=Context(**value["context"]),
                route_ref=tuple(value["route_ref"]), exchange_ref=tuple(value["exchange_ref"]),
                token_ids=tuple(value["token_ids"]),
                valid_slots=tuple(tuple(r) for r in value["valid_slots"]),
                inverse_map=tuple(tuple(r) for r in value["inverse_map"]),
                hidden_size=value["hidden_size"], gradient_boundary=value["gradient_boundary"],
                policy_hash=value["policy_hash"],
            )
        except (TypeError, KeyError) as exc:
            raise ContractError("SCHEMA_MISMATCH", "malformed Plan") from exc

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    @property
    def order_hash(self):
        # Logical order does not contain transport arrival/physical packed order.
        return digest({"tokens": self.token_ids, "valid": self.valid_slots, "policy": self.policy_hash})


@dataclass(frozen=True)
class SavedForward:
    plan_json: str
    fingerprint: str

    @classmethod
    def capture(cls, plan, context):
        plan.validate(context)
        return cls(canonical_json(plan.to_dict()), plan.fingerprint)

    def restore(self, context, expected_fingerprint):
        try:
            value = json.loads(self.plan_json)
        except (TypeError, ValueError) as exc:
            raise ContractError("CORRUPT_METADATA", "saved plan JSON") from exc
        require(digest(value) == self.fingerprint == expected_fingerprint,
                "CORRUPT_METADATA", "forward fingerprint mismatch")
        plan = Plan.from_dict(value)
        plan.validate(context)
        return plan


def production_provider(*args, **kwargs):
    raise ContractError("UNSUPPORTED_CAPABILITY", "T01 is a recorded-reference draft, not a production provider")
