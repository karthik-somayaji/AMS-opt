# MCQs for Knowledge Graph Evaluation

Tests whether a small (~8B) model can answer circuit KG questions better when provided with KGs from similar circuits.

## Diff Amps

| Circuit | Difficulty | Question |
|---------|-----------|----------|
| 71 | Easy | Which performance metrics does the IB1 bias current source influence? Direct substructure→performance lookup. |
| 75 | Easy | Which functional substructure exists in this circuit? Factual recall from KG node names. |
| 77 | Medium | Which parameter has an ambiguous relationship with all four metrics? Requires scanning multiple parameter→performance edges. |
| 80 | Medium | Which substructure does NOT influence Gain? Requires cross-checking substructure→performance links with negation. |
| 69 | Hard | Which parameter to tune for gain without hurting UGF? Requires reasoning across multiple parameter→metric relationships simultaneously. |
