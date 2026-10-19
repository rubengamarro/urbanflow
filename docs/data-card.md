# Data card

Provider: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), official yellow taxi Parquet and taxi zone CSV. Default snapshot: January–February 2024. Manifest fields include URL, source page, retrieval UTC timestamp, bytes, ETag and SHA-256. ETag is metadata, not a checksum. Initial download is trust-on-first-use over HTTPS; subsequent cached copies must match their recorded hash.

Vendor-supplied records can contain errors or incomplete submissions. TLC describes monthly publication, typically with a two-month delay. API availability does not establish data freshness.

## Eligibility

Required trip columns: `tpep_pickup_datetime`, `PULocationID`. Required warehouse lookup fields: `LocationID`, `Zone`, `Borough`, `service_zone`. Other fare-schema changes are irrelevant to this target.

Keep parseable pickups within their source calendar month, integral geographic zones 1–263 and valid unambiguous NYC timestamps. Exclude unknown/outside IDs 264–265. Fare, distance, passengers, payment and dropoff are not filters because they do not define whether a valid pickup occurred. Preserve identical pickup timestamp/zone pairs: the projection has no unique trip ID. Rejection counts are recorded by month.

## Hourly contract

| Column | Type | Meaning |
|---|---|---|
| hour | Timestamp, naive UTC convention | Hour start |
| zone_id | Integer 1–263 | Geographic taxi zone |
| demand | Nonnegative int32 | Eligible pickups |

Exactly one row per scoped zone/hour. Months must be unique and consecutive. Grid hours come from complete source months; absent count rows are zeros, absent months are errors. Vendor outages remain unobservable. TLC Parquet uses microsecond timestamps; Spark fixtures use that resolution rather than unsupported Parquet nanoseconds.

Spring DST has 23 elapsed hours; nonexistent timestamps are rejected. Ambiguous autumn timestamps cause an explicit failure because no offset is available to recover two distinct UTC hours. Do not include the fall-back month in this version.

## Measured reference quality

| Month | Raw rows | Eligible | Rejected |
|---|---:|---:|---:|
| 2024-01 | 2,964,624 | 2,952,588 | 12,036 |
| 2024-02 | 3,007,526 | 2,996,418 | 11,108 |

The panel has 378,720 rows, 263 zones, and 60.5569% zero counts. Exact hashes and quality outputs are in [quality.json](../reports/reference/quality.json). These numbers come from actual processing, not hardcoded model inputs.

The target is observed yellow taxi activity, not latent/unserved demand or all mobility demand. Historical activity reflects supply, access and socioeconomic disparities; zero pickups are not proof that a neighborhood has no mobility needs.
