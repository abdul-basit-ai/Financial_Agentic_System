# ADR-002: Neo4j AuraDB vs Self-Hosted Neo4j

**Status:** Decided (self-hosted for Phases 1–16; revisit at Phase 17 IaC)
**Date:** 2026-09-07
**Feeds:** Phase 17 (Terraform modules), Phase 18 (K8s deployment)

## Context

The knowledge graph (Phases 3, 6, 10) needs a Neo4j deployment target for local
development now and for cloud deployment in Phase 17/18. Two options:

1. **Neo4j AuraDB** (managed: Free / Professional / Enterprise tiers)
2. **Self-hosted Neo4j** (docker-compose locally; Neo4j on K8s via the
   official Helm chart or stateful manifests in the cloud)

## Decision

**Self-hosted.** docker-compose (`neo4j:5`, auth `neo4j/password`) is the
dev target; the Phase 18 K8s manifests will run the same container image.

## Rationale

| Factor | AuraDB | Self-hosted |
|---|---|---|
| Dev cost | Free tier: 200k nodes cap — our graph is ~10k records * 20+ nodes/record at full load; **cap would be hit mid-Phase-3** | Free (local machine) |
| Prod cost (Phases 17–18) | AuraDB Professional ~$500+/mo baseline | One more StatefulSet + 50Gi PVC: near-zero marginal cost on an existing cluster |
| Latency | Cross-region round-trip per query | Same-cluster network (~ms) |
| Ops burden | Zero (managed backups, upgrades) | Backups + upgrades are ours (k8s CronJob for `neo4j-admin dump`) |
| Vector index | AuraDB supports vector indexes only on paid tiers | Full support in Community via docker image |
| Data gravity | DVC/evaluation pipeline already Postgres+Redis colocated | Graph colocated with the rest — one network policy story |

Deciding factors: the free-tier node cap is a hard blocker for the full FinQA
load, paid tiers exceed the project's cost envelope for a portfolio project,
and colocation with pgvector/Redis keeps the local-first principle (Design
Principle 4) intact.

## Consequences

- **Positive:** no recurring cost; full vector-index support; one `values.yaml`
  governs graph sizing alongside the other stores; dev/prod parity via the
  same image.
- **Negative:** backup/upgrade responsibility moves to us (mitigated: scheduled
  dumps in Phase 18, single-node deployment acceptable at this scale).
- **Revisit trigger:** if the project is deployed for real multi-user traffic
  where managed backups/HA matter more than cost, revisit AuraDB
  Professional. This ADR should be re-evaluated, not silently extended.

## References

- Graph schema: `docs/GRAPH_SCHEMA.md`
- Loader: `ingestion/graph_loader.py` (schema constraints + integrity checks)
- Design principle: `project_goals.md` — "Local-first, cloud-later"
