# Data Quality Summary

Explicit date filter: `month >= 2015-10-01`.
Processed coverage: 2015-10-01 to 2026-05-01.
Retained Trusts/providers: 500.
Retained specialties: 24.
Retained months: 128.
Retained Trust-specialty series: 4992.
Series excluded for insufficient history: 1453.
Missing months inserted during continuity completion: 46753.
Warning-level audit rows: 327.

Identifier harmonisation trims codes and normalises names only. It does not automatically merge organisations.
Stock waiting-list variables may be forward-filled only for missing publication months and are flagged. Activity flow variables are not forward-filled.
Processed rows retain source publication provenance through source ZIP, CSV, URL, publication month, and source row-count audit columns.