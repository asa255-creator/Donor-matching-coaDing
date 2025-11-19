#!/usr/bin/env python3
"""
Diagnostic tool for analyzing donor matching confidence distributions.

This script helps identify issues with the matching model by:
1. Analyzing the distribution of confidence scores
2. Identifying cases where obvious matches are missed
3. Finding high-confidence false positives
4. Providing insights for model improvement

Usage:
    python3 diagnostic_matching.py <kref_csv> <fec_csv> <model_weights_json>
"""

import sys
import json
import csv
import random
from collections import defaultdict
import math

# Copy similarity functions from Dedupe
def jaro_winkler(a, b):
    a = (a or "").upper()
    b = (b or "").upper()
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    if a == b: return 1.0

    m_dist = max(0, max(len(a), len(b))//2 - 1)
    a_m = [False]*len(a)
    b_m = [False]*len(b)
    matches = 0

    for i in range(len(a)):
        start = max(0, i - m_dist)
        end = min(i + m_dist + 1, len(b))
        for j in range(start, end):
            if b_m[j] or a[i] != b[j]:
                continue
            a_m[i] = True
            b_m[j] = True
            matches += 1
            break

    if matches == 0: return 0.0

    t = 0
    k = 0
    for i in range(len(a)):
        if not a_m[i]:
            continue
        while not b_m[k]:
            k += 1
        if a[i] != b[k]:
            t += 1
        k += 1
    t = t / 2.0

    j = (matches/len(a) + matches/len(b) + (matches - t)/matches) / 3.0
    l = 0
    while l < 4 and l < len(a) and l < len(b) and a[l] == b[l]:
        l += 1

    return j + l*0.1*(1.0 - j)

def levenshtein(s1, s2):
    s1 = (s1 or "").upper()
    s2 = (s2 or "").upper()
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def soundex(s):
    s = (s or "").upper()
    if not s:
        return "0000"
    s = "".join(c for c in s if c.isalpha())
    if not s:
        return "0000"

    soundex_map = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }

    result = s[0]
    prev_code = soundex_map.get(s[0], "0")
    for char in s[1:]:
        code = soundex_map.get(char, "0")
        if code != "0" and code != prev_code:
            result += code
        prev_code = code
        if len(result) == 4:
            break

    result = (result + "000")[:4]
    return result

def enhanced_similarity(str1, str2, field_type):
    """Simplified version without associations for diagnostics."""
    str1 = (str1 or "").strip().upper()
    str2 = (str2 or "").strip().upper()

    if str1 == str2:
        return 1.0

    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    # Jaro-Winkler
    jw_score = jaro_winkler(str1, str2)

    # Levenshtein
    max_len = max(len(str1), len(str2))
    lev_distance = levenshtein(str1, str2)
    lev_score = 1.0 - (lev_distance / max_len) if max_len > 0 else 1.0

    # Token-based
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())
    token_jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2) if (tokens1 or tokens2) else 0.0

    if field_type in ["name", "first", "last"]:
        phonetic_match = 1.0 if soundex(str1) == soundex(str2) else 0.0
        base_score = jw_score * 0.35 + lev_score * 0.25 + token_jaccard * 0.2 + phonetic_match * 0.2
        return max(base_score, phonetic_match * 0.8)
    elif field_type == "address":
        return jw_score * 0.4 + lev_score * 0.3 + token_jaccard * 0.3
    else:
        return jw_score * 0.5 + lev_score * 0.3 + token_jaccard * 0.2

def features(a, b):
    """Calculate 17 features for a pair."""
    # Full name
    full_a = (a.get("DonorFirst","") + " " + a.get("DonorLast","")).strip()
    full_b = (b.get("DonorFirst","") + " " + b.get("DonorLast","")).strip()
    f0 = enhanced_similarity(full_a, full_b, "name")

    # Last name
    f1 = enhanced_similarity(a.get("DonorLast",""), b.get("DonorLast",""), "last")

    # Address
    f2 = enhanced_similarity(a.get("Address1",""), b.get("Address1",""), "address")

    # Employer
    f3 = enhanced_similarity(a.get("Employer",""), b.get("Employer",""), "employer")

    # Occupation
    f4 = enhanced_similarity(a.get("Occupation",""), b.get("Occupation",""), "occupation")

    # Geographic
    zip_a = (a.get("Zip","") or "").strip()[:5]
    zip_b = (b.get("Zip","") or "").strip()[:5]
    f5 = 1.0 if (zip_a and zip_b and zip_a == zip_b) else 0.0

    city_a = (a.get("City","") or "").strip().upper()
    city_b = (b.get("City","") or "").strip().upper()
    f6 = 1.0 if (city_a and city_b and city_a == city_b) else 0.0

    # Interactions
    f7 = f0 * f1
    f8 = f0 * f2
    f9 = f0 * f3
    f10 = f0 * f4
    f11 = f1 * f2
    f12 = f1 * f3
    f13 = f1 * f4
    f14 = f2 * f3
    f15 = f2 * f4
    f16 = f3 * f4

    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]

def predict(a, b, weights):
    """Calculate match probability."""
    x = features(a, b)
    z = weights[-1]  # bias
    for i in range(len(x)):
        z += weights[i] * x[i]
    return 1.0 / (1.0 + math.exp(-z))

def load_csv(filename):
    """Load CSV file."""
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def analyze_distribution(data, weights, sample_size=1000):
    """Analyze the confidence score distribution."""
    print("\n" + "="*60)
    print("CONFIDENCE SCORE DISTRIBUTION ANALYSIS")
    print("="*60)

    # Sample pairs
    print(f"\nSampling {sample_size} random pairs...")
    scores = []
    examples_low = []  # < 0.3
    examples_mid = []  # 0.4-0.6
    examples_high = []  # > 0.7

    for _ in range(sample_size):
        a = random.choice(data)
        b = random.choice(data)

        if a == b:
            continue

        score = predict(a, b, weights)
        scores.append(score)

        # Collect examples
        if score < 0.3 and len(examples_low) < 5:
            examples_low.append((a, b, score))
        elif 0.4 <= score <= 0.6 and len(examples_mid) < 5:
            examples_mid.append((a, b, score))
        elif score > 0.7 and len(examples_high) < 5:
            examples_high.append((a, b, score))

    # Statistics
    scores.sort()
    n = len(scores)
    mean = sum(scores) / n
    median = scores[n//2]

    # Distribution buckets
    buckets = {
        "0.0-0.1": 0,
        "0.1-0.2": 0,
        "0.2-0.3": 0,
        "0.3-0.4": 0,
        "0.4-0.5": 0,
        "0.5-0.6": 0,
        "0.6-0.7": 0,
        "0.7-0.8": 0,
        "0.8-0.9": 0,
        "0.9-1.0": 0
    }

    for score in scores:
        bucket_idx = min(int(score * 10), 9)
        bucket_key = f"{bucket_idx/10:.1f}-{(bucket_idx+1)/10:.1f}"
        buckets[bucket_key] += 1

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Mean:   {mean:.3f}")
    print(f"  Median: {median:.3f}")
    print(f"  Min:    {scores[0]:.3f}")
    print(f"  Max:    {scores[-1]:.3f}")

    print(f"\nDistribution (expecting bimodal - peaks at <0.3 and >0.7):")
    max_count = max(buckets.values())
    for bucket, count in buckets.items():
        pct = count / n * 100
        bar_len = int(count / max_count * 50)
        bar = "█" * bar_len
        print(f"  {bucket}: {bar} {count:4d} ({pct:5.1f}%)")

    # Check for bimodal
    low_conf = sum(1 for s in scores if s < 0.3)
    mid_conf = sum(1 for s in scores if 0.3 <= s <= 0.7)
    high_conf = sum(1 for s in scores if s > 0.7)

    print(f"\nBimodal Analysis:")
    print(f"  Low confidence  (<0.3): {low_conf:4d} ({low_conf/n*100:5.1f}%)")
    print(f"  Mid confidence (0.3-0.7): {mid_conf:4d} ({mid_conf/n*100:5.1f}%)")
    print(f"  High confidence (>0.7): {high_conf:4d} ({high_conf/n*100:5.1f}%)")

    if mid_conf > low_conf and mid_conf > high_conf:
        print("\n  ⚠️  WARNING: Distribution is NOT bimodal!")
        print("      Most scores are in the uncertain middle range.")
        print("      This suggests the model is not confident in its predictions.")
    else:
        print("\n  ✓  Distribution appears bimodal (good!)")

    # Print example pairs
    print("\n" + "="*60)
    print("EXAMPLE PAIRS")
    print("="*60)

    def print_pair(a, b, score, label):
        print(f"\n{label} (confidence: {score:.3f})")
        print(f"  Person A: {a.get('DonorFirst','')} {a.get('DonorLast','')} | {a.get('Address1','')} | {a.get('City','')} {a.get('Zip','')}")
        print(f"  Person B: {b.get('DonorFirst','')} {b.get('DonorLast','')} | {b.get('Address1','')} | {b.get('City','')} {b.get('Zip','')}")

        # Show feature breakdown
        feat = features(a, b)
        print(f"  Features: name={feat[0]:.2f}, last={feat[1]:.2f}, addr={feat[2]:.2f}, emp={feat[3]:.2f}, occ={feat[4]:.2f}, zip={feat[5]:.0f}, city={feat[6]:.0f}")

    print("\n--- LOW CONFIDENCE EXAMPLES (should be clear non-matches) ---")
    for a, b, score in examples_low:
        print_pair(a, b, score, "Low confidence")

    print("\n--- MID CONFIDENCE EXAMPLES (uncertain) ---")
    for a, b, score in examples_mid:
        print_pair(a, b, score, "Mid confidence")

    print("\n--- HIGH CONFIDENCE EXAMPLES (should be clear matches) ---")
    for a, b, score in examples_high:
        print_pair(a, b, score, "High confidence")

def find_obvious_misses(data, weights, threshold=0.7):
    """Find pairs that should obviously match but don't."""
    print("\n" + "="*60)
    print("SEARCHING FOR OBVIOUS MISSES")
    print("="*60)

    # Look for same address + similar name
    print("\nLooking for pairs with:")
    print("  - Exact same address")
    print("  - Similar last name (soundex match)")
    print("  - But confidence < 0.7")

    address_index = defaultdict(list)
    for donor in data:
        addr = (donor.get("Address1","") or "").strip().upper()
        if addr:
            address_index[addr].append(donor)

    misses = []
    checked = 0

    for addr, donors in address_index.items():
        if len(donors) < 2:
            continue

        for i in range(len(donors)):
            for j in range(i+1, len(donors)):
                a = donors[i]
                b = donors[j]

                # Check if last names sound similar
                last_a = a.get("DonorLast","")
                last_b = b.get("DonorLast","")

                if soundex(last_a) == soundex(last_b):
                    score = predict(a, b, weights)
                    checked += 1

                    if score < threshold:
                        misses.append((a, b, score))

    print(f"\nChecked {checked} pairs with same address + phonetic last name match")
    print(f"Found {len(misses)} with confidence < {threshold}")

    if misses:
        print(f"\nShowing first 10 misses:")
        for a, b, score in misses[:10]:
            print(f"\n  Confidence: {score:.3f} (should be >0.7!)")
            print(f"    Person A: {a.get('DonorFirst','')} {a.get('DonorLast','')} | {a.get('Address1','')}")
            print(f"    Person B: {b.get('DonorFirst','')} {b.get('DonorLast','')} | {b.get('Address1','')}")
            feat = features(a, b)
            print(f"    Features: name={feat[0]:.2f}, last={feat[1]:.2f}, addr={feat[2]:.2f}")
    else:
        print("\n  ✓ No obvious misses found!")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 diagnostic_matching.py <kref_csv> <fec_csv> <model_weights_json>")
        sys.exit(1)

    kref_file = sys.argv[1]
    fec_file = sys.argv[2]
    model_file = sys.argv[3]

    print("Loading data...")
    kref_data = load_csv(kref_file)
    fec_data = load_csv(fec_file)
    combined = kref_data + fec_data

    print(f"Loaded {len(kref_data)} KREF records")
    print(f"Loaded {len(fec_data)} FEC records")
    print(f"Total: {len(combined)} records")

    print("\nLoading model weights...")
    with open(model_file, 'r') as f:
        model_data = json.load(f)
        weights = model_data.get("weights")

    if not weights:
        print("ERROR: No weights found in model file!")
        sys.exit(1)

    print(f"Loaded model with {len(weights)} weights")

    # Run diagnostics
    analyze_distribution(combined, weights, sample_size=2000)
    find_obvious_misses(combined, weights, threshold=0.7)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
