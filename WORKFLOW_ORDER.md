# Donor Matching System - Workflow Order

**CRITICAL: This order MUST be maintained. Breaking it destroys the entire system.**

## Complete Workflow

### Phase 1: Data Import (Donations Finder)
1. **Upload FEC Reports** (KREF Tools → Add downloaded FEC reports)
   - Imports federal donation data
   - Goes into `FEC_Exports` sheet
   - File: `Donations Finder`

2. **Download KREF Information** (KREF Tools → Fetch donor info from KREF)
   - Fetches local donation data from KREF database
   - Goes into `KREF_Exports` sheet
   - File: `Donations Finder`

### Phase 2: Donor Deduplication (Dedupe via Terminal)
3. **Merge and Match Donors** (Donor Tools → Run locally: prepare job and show Terminal command)
   - Runs LOCALLY via Python in Terminal
   - Combines FEC_Exports + KREF_Exports
   - Creates training pairs for manual labeling
   - Trains machine learning model (logistic regression)
   - Assigns DonorIDs to matched donors using Union-Find clustering
   - Creates merged dataset with DonorIDs
   - File: `Dedupe` (Python script embedded in GAS)

### Phase 3: Campaign Deputy Matching (Dedupe)
4. **Match Donors to Campaign Deputy** (Donor Tools → Match Donors to Campaign Deputy)
   - Takes merged donor data (with DonorIDs)
   - Matches against Campaign Deputy PersonIDs
   - Creates `CD_To_Upload` sheet
   - File: `Dedupe`

### Phase 4: Geocoding & District Assignment (SortingAndLabeling)
5. **Improve Geocode Cache** (Donor Tools → Improve Geocode Cache)
   - Geocodes addresses in `CD_To_Upload`
   - Updates `_GeocodeCache` sheet
   - Handles pending addresses with progressive normalization
   - File: `SortingAndLabeling`

6. **Match Addresses to Districts** (Donor Tools → Match Addresses to Districts)
   - Uses geocoded coordinates
   - Assigns Federal Districts, State Senate/House Districts, Counties
   - Updates `CD_To_Upload` with district information
   - File: `SortingAndLabeling`

## Data Flow Dependencies

```
FEC_Exports ─┐
             ├──> [Terminal Deduplication] ──> Merged Data with DonorIDs ──> [CD Matching] ──> CD_To_Upload ──> [Geocoding] ──> [District Matching]
KREF_Exports─┘
```

## File Responsibilities

| File | Purpose |
|------|---------|
| **Donations Finder** | Import FEC and KREF data |
| **Dedupe** | Donor deduplication, training, Campaign Deputy matching |
| **SortingAndLabeling** | Geocoding and district assignment |
| **On-open** | Menu setup and sheet initialization |

## CRITICAL RULES

1. **FEC upload code MUST be in Donations Finder** (not Dedupe)
2. **Never modify Dedupe unless changing deduplication/matching logic**
3. **Geocoding code stays in SortingAndLabeling**
4. **Terminal deduplication MUST happen before CD matching**
5. **Geocoding MUST happen after CD matching creates CD_To_Upload**

## Common Mistakes

❌ **Don't put FEC upload in Dedupe** - breaks file organization and risks syntax errors
❌ **Don't run CD matching before deduplication** - no DonorIDs exist yet
❌ **Don't geocode before CD_To_Upload exists** - nothing to geocode

## Training Data Requirements

For 90-95% matching accuracy:
- **Initial**: 500-1,000 labeled pairs (50/50 match/non-match ratio)
- **After active learning**: 1,500-3,000 total pairs
- Use "Continue training: add more labeled pairs" to expand dataset
