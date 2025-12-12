#!/usr/bin/env python3
"""
Evolutionary Blocking Rule Optimization
Continuously evolves blocking rules to improve matching performance.
Press Ctrl+C to stop and save results.
"""

import sys
import csv
import json
import math
import random
import signal
import urllib.request
from collections import defaultdict
import time

# Global flag for graceful shutdown
should_stop = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global should_stop
    print("\n" + "="*60)
    print("STOPPING - Saving best rules...")
    print("="*60)
    should_stop = True

signal.signal(signal.SIGINT, signal_handler)

# ============================================
# SOUNDEX IMPLEMENTATION
# ============================================
def soundex(name):
    """Standard Soundex algorithm"""
    if not name:
        return ""
    name = name.upper()
    s = name[0]

    # Soundex digit mapping
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    for c in name[1:]:
        d = mapping.get(c, '')
        if d and d != s[-1]:
            s += d
        elif c not in 'AEIOUYHW' and not d:
            pass

    s += '000'
    return s[:4]

# ============================================
# BLOCKING KEY GENERATION
# ============================================
def generate_blocking_key_from_rule(donor, rule):
    """Generate blocking key for a donor using a rule"""
    fields = rule.get('fields', [])
    transforms = rule.get('transforms', [])
    separator = rule.get('separator', '|')

    parts = []
    for i, field in enumerate(fields):
        value = donor.get(field, '').strip()
        if not value:
            return None  # Missing required field

        # Apply transform
        transform = transforms[i] if i < len(transforms) else 'none'

        if transform == 'soundex':
            value = soundex(value)
        elif transform == 'lower':
            value = value.lower()
        elif transform == 'first2':
            value = value[:2].upper()
        elif transform == 'first3':
            value = value[:3].upper()
        elif transform == 'first4':
            value = value[:4].upper()
        elif transform == 'first5':
            value = value[:5].upper()
        elif transform == 'first_token':
            tokens = value.split()
            value = tokens[0].upper() if tokens else ''
        elif transform == 'none':
            value = value.upper()

        if not value:
            return None

        parts.append(value)

    return separator.join(parts)

# ============================================
# FEATURE COMPUTATION (same as training)
# ============================================
def jaro_winkler(s1, s2):
    """Jaro-Winkler similarity"""
    if not s1 or not s2:
        return 0.0

    # Jaro distance
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    match = 0
    hash_s1 = [0] * len1
    hash_s2 = [0] * len2

    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    if match == 0:
        return 0.0

    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1
    t /= 2

    jaro = (match / len1 + match / len2 + (match - t) / match) / 3.0

    # Winkler modification
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + (prefix * 0.1 * (1.0 - jaro))

def features(a, b):
    """Compute feature vector for two donors"""
    def safe_jw(s1, s2):
        return jaro_winkler(str(s1).lower(), str(s2).lower())

    # Single field features
    feat_name = safe_jw(a.get('DonorFirst', ''), b.get('DonorFirst', ''))
    feat_last = safe_jw(a.get('DonorLast', ''), b.get('DonorLast', ''))
    feat_addr = safe_jw(a.get('Address1', ''), b.get('Address1', ''))
    feat_emp = safe_jw(a.get('Employer', ''), b.get('Employer', ''))
    feat_occ = safe_jw(a.get('Occupation', ''), b.get('Occupation', ''))

    # Exact matches
    zip_a = str(a.get('Zip', '')).strip()[:5]
    zip_b = str(b.get('Zip', '')).strip()[:5]
    feat_zip = 1.0 if (zip_a and zip_b and zip_a == zip_b) else 0.0

    city_a = str(a.get('City', '')).strip().lower()
    city_b = str(b.get('City', '')).strip().lower()
    feat_city = 1.0 if (city_a and city_b and city_a == city_b) else 0.0

    # Two-field products
    feat_name_last = feat_name * feat_last
    feat_name_addr = feat_name * feat_addr
    feat_name_emp = feat_name * feat_emp
    feat_name_occ = feat_name * feat_occ
    feat_last_addr = feat_last * feat_addr
    feat_last_emp = feat_last * feat_emp
    feat_last_occ = feat_last * feat_occ
    feat_addr_emp = feat_addr * feat_emp
    feat_addr_occ = feat_addr * feat_occ
    feat_emp_occ = feat_emp * feat_occ

    return [
        feat_name, feat_last, feat_addr, feat_emp, feat_occ, feat_zip, feat_city,
        feat_name_last, feat_name_addr, feat_name_emp, feat_name_occ,
        feat_last_addr, feat_last_emp, feat_last_occ,
        feat_addr_emp, feat_addr_occ, feat_emp_occ
    ]

# ============================================
# EXACT MATCH CONDENSATION
# ============================================
def condense_exact_matches(donors):
    """Condense exact name+address duplicates to reduce dataset size"""
    exact_map = {}
    uf_parent = {}
    uf_rank = {}

    def uf_find(x):
        if x not in uf_parent:
            uf_parent[x] = x
            uf_rank[x] = 0
        if uf_parent[x] != x:
            uf_parent[x] = uf_find(uf_parent[x])
        return uf_parent[x]

    def uf_union(x, y):
        px, py = uf_find(x), uf_find(y)
        if px == py:
            return
        if uf_rank[px] < uf_rank[py]:
            uf_parent[px] = py
        elif uf_rank[px] > uf_rank[py]:
            uf_parent[py] = px
        else:
            uf_parent[py] = px
            uf_rank[px] += 1

    for idx, donor in enumerate(donors):
        first = (donor.get("DonorFirst") or "").strip().upper()
        last = (donor.get("DonorLast") or "").strip().upper()
        addr = (donor.get("Address1") or "").strip().upper()
        city = (donor.get("City") or "").strip().upper()
        state = (donor.get("State") or "").strip().upper()
        zip_code = (donor.get("Zip") or "").strip()

        if first and last and addr:
            key = (first, last, addr, city, state, zip_code)
            if key in exact_map:
                uf_union(idx, exact_map[key])
            else:
                exact_map[key] = idx

    # Keep one representative per group
    representatives = {}
    for idx in range(len(donors)):
        root = uf_find(idx)
        if root not in representatives:
            representatives[root] = donors[idx]

    condensed = list(representatives.values())
    print(f"Condensed {len(donors)} donors → {len(condensed)} unique donors")
    return condensed

# ============================================
# RULE EVALUATION
# ============================================
def evaluate_blocking_rule(rule, donors, w, comparison_budget):
    """
    Evaluate a blocking rule by running limited comparisons and counting matches.

    Returns: number of matches found within comparison budget
    """
    # Apply blocking rule to create blocks
    blocks = {}
    for donor in donors:
        key = generate_blocking_key_from_rule(donor, rule)
        if key:
            if key not in blocks:
                blocks[key] = []
            blocks[key].append(donor)

    if not blocks:
        return 0.0

    # Calculate total available comparisons
    total_pairs_available = sum(len(block) * (len(block) - 1) // 2 for block in blocks.values())

    if total_pairs_available == 0:
        return 0.0

    # Proportionally sample from each block
    pairs_to_compare = []
    for block_key, block_donors in blocks.items():
        block_size = len(block_donors)
        if block_size < 2:
            continue

        block_pairs_available = block_size * (block_size - 1) // 2
        # Allocate comparisons proportionally
        num_from_block = int(comparison_budget * block_pairs_available / total_pairs_available)

        # Generate all pairs in block
        block_pairs = []
        for i in range(block_size):
            for j in range(i + 1, block_size):
                block_pairs.append((block_donors[i], block_donors[j]))

        # Sample from this block
        if num_from_block < len(block_pairs):
            sampled = random.sample(block_pairs, num_from_block)
        else:
            sampled = block_pairs

        pairs_to_compare.extend(sampled)

    # Limit to budget (in case of rounding)
    if len(pairs_to_compare) > comparison_budget:
        pairs_to_compare = random.sample(pairs_to_compare, comparison_budget)

    # Score pairs with ML model and count matches
    threshold = 0.7
    matches_found = 0

    for a, b in pairs_to_compare:
        x = features(a, b)
        z = w[-1]  # bias
        for t in range(len(x)):
            z += w[t] * x[t]
        p = 1.0 / (1.0 + math.exp(-z))
        if p >= threshold:
            matches_found += 1

    return float(matches_found)

# ============================================
# RULE MUTATION AND CROSSOVER
# ============================================
def mutate_rule(rule):
    """Create a mutated version of a rule"""
    new_rule = {
        'name': rule['name'] + '_mut',
        'fields': rule['fields'][:],
        'transforms': rule['transforms'][:],
        'separator': rule['separator'],
        'cohort': rule['cohort'],
        'score': 0.0
    }

    mutation_type = random.choice(['change_transform', 'add_field', 'remove_field'])

    all_fields = ['DonorFirst', 'DonorLast', 'Address1', 'City', 'State', 'Zip']
    all_transforms = ['soundex', 'lower', 'first2', 'first3', 'first4', 'first5', 'first_token', 'none']

    if mutation_type == 'change_transform' and new_rule['transforms']:
        idx = random.randint(0, len(new_rule['transforms']) - 1)
        new_rule['transforms'][idx] = random.choice(all_transforms)

    elif mutation_type == 'add_field' and len(new_rule['fields']) < 3:
        available = [f for f in all_fields if f not in new_rule['fields']]
        if available:
            new_field = random.choice(available)
            new_rule['fields'].append(new_field)
            new_rule['transforms'].append(random.choice(all_transforms))

    elif mutation_type == 'remove_field' and len(new_rule['fields']) > 1:
        idx = random.randint(0, len(new_rule['fields']) - 1)
        del new_rule['fields'][idx]
        del new_rule['transforms'][idx]

    # Update name
    new_rule['name'] = ' + '.join([
        f"{t}({f})" if t != 'none' else f
        for f, t in zip(new_rule['fields'], new_rule['transforms'])
    ])

    return new_rule

def crossover_rules(rule1, rule2):
    """Create a new rule by combining two rules"""
    # Take fields from both parents
    all_fields = list(set(rule1['fields'] + rule2['fields']))
    if len(all_fields) > 3:
        all_fields = random.sample(all_fields, 3)

    # Randomly pick transforms
    transforms = []
    all_transforms = ['soundex', 'lower', 'first2', 'first3', 'first4', 'first5', 'first_token', 'none']
    for field in all_fields:
        if field in rule1['fields']:
            idx = rule1['fields'].index(field)
            transforms.append(rule1['transforms'][idx])
        elif field in rule2['fields']:
            idx = rule2['fields'].index(field)
            transforms.append(rule2['transforms'][idx])
        else:
            transforms.append(random.choice(all_transforms))

    name = ' + '.join([
        f"{t}({f})" if t != 'none' else f
        for f, t in zip(all_fields, transforms)
    ])

    return {
        'name': name,
        'fields': all_fields,
        'transforms': transforms,
        'separator': '|',
        'cohort': rule1['cohort'],  # Inherit cohort from first parent
        'score': 0.0
    }

# ============================================
# RULE SIMILARITY
# ============================================
def calculate_rule_similarity(rule1, rule2):
    """Calculate Jaccard similarity between two rules"""
    fields1 = set(rule1.get('fields', []))
    fields2 = set(rule2.get('fields', []))

    if not fields1 and not fields2:
        return 1.0
    if not fields1 or not fields2:
        return 0.0

    intersection = len(fields1 & fields2)
    union = len(fields1 | fields2)

    return intersection / union if union > 0 else 0.0

# ============================================
# COHORT SELECTION
# ============================================
def select_cohort_survivors(cohorts):
    """
    Select top 5 rules from each cohort with diversity constraints.

    Cohort 1: Top 5 by raw score (no penalty)
    Cohort 2-5: Top 5 by adjusted score (penalized for similarity to previous cohorts)
    """
    all_survivors = []

    for cohort_idx, cohort_rules in enumerate(cohorts):
        if cohort_idx == 0:
            # Cohort 1: Pure performance
            cohort_rules.sort(key=lambda x: x[0], reverse=True)
            survivors = cohort_rules[:5]
            all_survivors.append(survivors)
            print(f"  Cohort 1: Top 5 by performance")
            for i, (score, rule) in enumerate(survivors, 1):
                print(f"    {i}. {score:.1f} matches - {rule['name']}")
        else:
            # Cohorts 2-5: Penalize similarity to previous cohorts
            adjusted_scores = []
            for score, rule in cohort_rules:
                # Calculate max similarity to any previous cohort
                max_similarity = 0.0
                for prev_cohort in all_survivors:
                    for (prev_score, prev_rule) in prev_cohort:
                        similarity = calculate_rule_similarity(rule, prev_rule)
                        max_similarity = max(max_similarity, similarity)

                # Diversity bonus (higher is better)
                diversity_bonus = (1 - max_similarity) * 30
                adjusted_score = score + diversity_bonus
                adjusted_scores.append((adjusted_score, score, rule))

            # Sort by adjusted score
            adjusted_scores.sort(key=lambda x: x[0], reverse=True)
            survivors = [(raw_score, rule) for (adj, raw_score, rule) in adjusted_scores[:5]]
            all_survivors.append(survivors)

            print(f"  Cohort {cohort_idx + 1}: Top 5 with diversity")
            for i, (score, rule) in enumerate(survivors, 1):
                print(f"    {i}. {score:.1f} matches - {rule['name']}")

    return all_survivors

# ============================================
# ENTRY POINT
# ============================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evolve blocking rules for donor matching')
    parser.add_argument('--rules', required=True, help='Path to blocking_rules.csv')
    parser.add_argument('--model-url', required=True, help='URL to fetch model data')
    parser.add_argument('--kref-url', required=True, help='URL to fetch KREF data')
    parser.add_argument('--fec-url', required=True, help='URL to fetch FEC data')

    args = parser.parse_args()

    print("="*60)
    print("EVOLUTIONARY BLOCKING RULE OPTIMIZATION")
    print("="*60)

    # Load initial rules from CSV
    print("\nLoading initial rules from CSV...")
    rules = []
    with open(args.rules, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rules.append({
                'name': row['Rule Name'],
                'fields': [f.strip() for f in row['Fields'].split(',')],
                'transforms': [t.strip() for t in row['Transforms'].split(',')],
                'separator': row['Separator'],
                'cohort': int(row['Cohort']),
                'score': float(row['Score'])
            })

    print(f"✓ Loaded {len(rules)} rules")

    # Load model data
    print("\nLoading model data...")
    try:
        with urllib.request.urlopen(args.model_url) as response:
            model_data = json.loads(response.read().decode('utf-8'))

        w = model_data.get('weights')
        if not w:
            print("✗ No model weights found. Train a model first.")
            sys.exit(1)

        print(f"✓ Loaded model with {len(w)} weights")

    except Exception as e:
        print(f"✗ Failed to load model data: {e}")
        sys.exit(1)

    # Load donor data
    print("\nLoading donor data from KREF and FEC...")
    donors = []

    try:
        # Load KREF
        with urllib.request.urlopen(args.kref_url) as response:
            kref_data = response.read().decode('utf-8')
            kref_reader = csv.DictReader(io.StringIO(kref_data))
            kref_donors = list(kref_reader)
            print(f"✓ Loaded {len(kref_donors)} KREF donors")

        # Load FEC
        with urllib.request.urlopen(args.fec_url) as response:
            fec_data = response.read().decode('utf-8')
            fec_reader = csv.DictReader(io.StringIO(fec_data))
            fec_donors = list(fec_reader)
            print(f"✓ Loaded {len(fec_donors)} FEC donors")

        donors = kref_donors + fec_donors
        print(f"✓ Total donors: {len(donors)}")

    except Exception as e:
        print(f"✗ Failed to load donor data: {e}")
        sys.exit(1)

    if not donors:
        print("\n✗ No donor data available. Exiting.")
        sys.exit(1)

    # Condense exact matches
    print("\nCondensing exact matches...")
    donors = condense_exact_matches(donors)

    # Evolution loop
    round_num = 0
    comparison_budget = 100000

    print(f"\nStarting evolution (budget: {comparison_budget:,} comparisons per rule)")
    print("Press Ctrl+C to stop and save results\n")

    while not should_stop:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")

        # Evaluate all rules
        print("\nEvaluating all 50 rules...")
        for rule in rules:
            rule['score'] = evaluate_blocking_rule(rule, donors, w, comparison_budget)

        # Group by cohort
        cohorts = [[] for _ in range(5)]
        for rule in rules:
            cohort_idx = rule['cohort'] - 1
            cohorts[cohort_idx].append((rule['score'], rule))

        # Select survivors (top 5 per cohort)
        survivors = select_cohort_survivors(cohorts)

        # Generate new rules (5 per cohort)
        new_rules = []
        for cohort_idx, cohort_survivors in enumerate(survivors):
            print(f"\nGenerating 5 new rules for Cohort {cohort_idx + 1}...")
            parent_rules = [rule for (score, rule) in cohort_survivors]

            for _ in range(5):
                if random.random() < 0.7:
                    # Mutation
                    parent = random.choice(parent_rules)
                    new_rule = mutate_rule(parent)
                else:
                    # Crossover
                    parent1, parent2 = random.sample(parent_rules, 2)
                    new_rule = crossover_rules(parent1, parent2)

                new_rule['cohort'] = cohort_idx + 1
                new_rules.append(new_rule)

        # Combine survivors and new rules
        rules = []
        for cohort_survivors in survivors:
            rules.extend([rule for (score, rule) in cohort_survivors])
        rules.extend(new_rules)

        print(f"\n✓ Round {round_num} complete. Population: {len(rules)} rules")
        time.sleep(0.1)  # Brief pause

    # Save results
    output_csv = 'blocking_rules_evolved.csv'
    print(f"\nSaving results to {output_csv}...")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rule Name', 'Fields', 'Transforms', 'Separator', 'Cohort', 'Score'])

        for rule in rules:
            writer.writerow([
                rule['name'],
                ', '.join(rule['fields']),
                ', '.join(rule['transforms']),
                rule['separator'],
                rule['cohort'],
                rule['score']
            ])

    print(f"✓ Saved {len(rules)} evolved rules to {output_csv}")
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
