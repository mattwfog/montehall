"""The declared colony is the runtime dependency graph, including absent organs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soccerviz.harness.engine import Specialist


@dataclass(frozen=True)
class OptionalDependency:
    name: str
    absent_behavior: str


@dataclass(frozen=True)
class OrganismManifest:
    name: str
    version: str
    required: tuple[str, ...]
    optional: tuple[OptionalDependency, ...]
    output_schema: str
    resources: str
    evaluation: str
    consumers: tuple[str, ...]
    available: bool
    unavailable_reason: str

    @classmethod
    def from_specialist(cls, spec: Specialist):
        return cls(
            spec.name,
            spec.version,
            tuple(spec.requires),
            tuple(OptionalDependency(*value) for value in spec.optional_requires),
            spec.output_schema,
            spec.resources,
            spec.evaluate,
            tuple(spec.consumers),
            spec.available,
            spec.unavailable_reason,
        )


class ColonyRegistry:
    """One declared graph drives validation, rendering and runtime ordering.

    Unavailable organs remain declared. A required unavailable input is an error;
    optional unavailable inputs have an explicit None payload and absence policy.
    Both required and optional edges participate in cycle detection.
    """

    def __init__(self, specialists):
        self.specialists = tuple(specialists)
        self.organisms = tuple(OrganismManifest.from_specialist(s) for s in self.specialists)
        self._by_name = {s.name: s for s in self.specialists}
        self.validate()

    def validate(self):
        if len(self._by_name) != len(self.specialists):
            raise ValueError("Colony specialist names must be unique")
        names = set(self._by_name)
        for organ in self.organisms:
            if not organ.name or not organ.version or not organ.output_schema:
                raise ValueError("Every organism needs a name, version and output schema")
            deps = (*organ.required, *(d.name for d in organ.optional))
            if len(set(deps)) != len(deps):
                raise ValueError(f"Duplicate required/optional dependency in {organ.name}")
            unknown = set(deps) - names
            if unknown:
                raise ValueError(f"Unknown dependencies for {organ.name}: {sorted(unknown)}")
            if any(not d.absent_behavior.strip() for d in organ.optional):
                raise ValueError(
                    f"Optional dependency needs declared absent behavior: {organ.name}"
                )
            if not organ.available and not organ.unavailable_reason.strip():
                raise ValueError(f"Unavailable organism needs a reason: {organ.name}")
            if organ.available:
                missing = [name for name in organ.required if not self._by_name[name].available]
                if missing:
                    raise ValueError(
                        f"Required dependencies unavailable for {organ.name}: {missing}"
                    )
        self._ordered_names(include_unavailable=True)

    def _ordered_names(self, include_unavailable=False):
        names = (
            set(self._by_name)
            if include_unavailable
            else {name for name, spec in self._by_name.items() if spec.available}
        )
        dependencies = {
            name: (
                set(self._by_name[name].requires)
                | {dep for dep, _behavior in self._by_name[name].optional_requires}
            )
            & names
            for name in names
        }
        result = []
        while dependencies:
            ready = sorted(name for name, deps in dependencies.items() if not deps)
            if not ready:
                raise ValueError(f"Colony dependency cycle: {sorted(dependencies)}")
            for name in ready:
                result.append(name)
                del dependencies[name]
            for deps in dependencies.values():
                deps.difference_update(ready)
        return result

    def execution_order(self):
        return tuple(self._by_name[name] for name in self._ordered_names())

    def manifest(self):
        return {
            "schema": "colony-manifest/v1",
            "organisms": [asdict(organ) for organ in sorted(self.organisms, key=lambda x: x.name)],
            "execution_order": self._ordered_names(),
            "unavailable": [
                {"name": organ.name, "reason": organ.unavailable_reason}
                for organ in sorted(self.organisms, key=lambda x: x.name)
                if not organ.available
            ],
        }
