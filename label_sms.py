"""
SMS Auto-Labeler for Silver Guard Training Data
------------------------------------------------
Input:  data.json (Phone Link SMS export)
        contacts.csv OR contacts.vcf (optional - Google/phone contacts export)
Output: output.json (labeled messages + review list)

DLT headers (XX-XXXXXX-X) = auto-labeled as ham (0)
Known contacts = auto-labeled as ham (0)
Unknown phone numbers = saved to review.json for manual labeling
"""

import json
import re
import csv
from collections import Counter
from pathlib import Path

INPUT_FILE = "data.json"
CONTACTS_FILE = "contacts.csv"  # or contacts.vcf
OUTPUT_FILE = "output.json"
REVIEW_FILE = "review.json"  # messages needing manual review

# DLT header pattern: XX-XXXXXX-X (telecom circle - entity - type)
DLT_PATTERN = re.compile(r'^[A-Z]{2}-[A-Z0-9]{5,7}-[STGP]$')

# Also match slightly different formats like JK-620040-P
DLT_PATTERN_ALT = re.compile(r'^[A-Z]{2}-[A-Z0-9]{4,8}-?[STGP]?$')


def normalize_phone(phone):
    """Normalize phone number for comparison (remove spaces, +91, etc.)"""
    if not phone:
        return None
    # Remove all non-digits
    digits = re.sub(r'\D', '', str(phone))
    # Remove leading 91 (India code) if present
    if len(digits) > 10 and digits.startswith('91'):
        digits = digits[2:]
    # Remove leading 0 if present
    if len(digits) > 10 and digits.startswith('0'):
        digits = digits[1:]
    # Return last 10 digits (standard Indian mobile)
    if len(digits) >= 10:
        return digits[-10:]
    return digits if digits else None


def load_contacts():
    """Load contacts from CSV (Google export) or VCF file."""
    contacts = {}  # normalized_phone -> name
    
    # Try CSV first (Google Contacts export)
    csv_path = Path(CONTACTS_FILE)
    if csv_path.exists():
        print(f"Loading contacts from {CONTACTS_FILE}...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('Name', '') or row.get('First Name', '') or row.get('Display Name', '')
                # Google CSV has Phone 1 - Value, Phone 2 - Value, etc.
                for key in row:
                    if 'phone' in key.lower() and row[key]:
                        norm = normalize_phone(row[key])
                        if norm:
                            contacts[norm] = name
        print(f"Loaded {len(contacts)} phone numbers from contacts")
        return contacts
    
    # Try VCF (vCard format)
    vcf_path = Path("contacts.vcf")
    if vcf_path.exists():
        print(f"Loading contacts from contacts.vcf...")
        current_name = ""
        with open(vcf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('FN:'):
                    current_name = line[3:]
                elif line.startswith('TEL'):
                    # TEL;TYPE=CELL:+919876543210 or TEL:+919876543210
                    phone = line.split(':')[-1]
                    norm = normalize_phone(phone)
                    if norm:
                        contacts[norm] = current_name
        print(f"Loaded {len(contacts)} phone numbers from contacts")
        return contacts
    
    print(f"No contacts file found ({CONTACTS_FILE} or contacts.vcf)")
    print("To use contact matching, export your contacts as CSV or VCF")
    return {}


def is_known_contact(address, contacts):
    """Check if phone number is in contact list."""
    norm = normalize_phone(address)
    if norm and norm in contacts:
        return contacts[norm]
    return None


def is_dlt_header(address):
    """Check if address is a proper DLT sender ID."""
    if not address:
        return False
    addr = str(address).strip()
    # Standard DLT format
    if DLT_PATTERN.match(addr):
        return True
    # Alternative formats (some carriers use slightly different patterns)
    if DLT_PATTERN_ALT.match(addr) and not addr.startswith('+'):
        return True
    return False


def is_phone_number(address):
    """Check if address is a phone number."""
    if not address:
        return False
    addr = str(address).strip()
    # Indian phone: +91XXXXXXXXXX or just digits
    if addr.startswith('+91') or addr.startswith('91'):
        return True
    if re.match(r'^\+?\d{10,15}$', addr):
        return True
    return False

def main():
    # Load contacts
    contacts = load_contacts()
    
    # Load input
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERROR: {INPUT_FILE} not found in current directory")
        print(f"Current directory: {Path.cwd()}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    # Handle nested format
    if isinstance(raw, dict) and 'messages' in raw:
        messages = raw['messages']
    elif isinstance(raw, list):
        messages = raw
    else:
        print("ERROR: Unrecognized JSON format")
        return
    
    print(f"Loaded {len(messages)} messages")
    
    # Filter out sent messages
    received = [m for m in messages if m.get('kind') != 'sent']
    print(f"Received messages (excluding sent): {len(received)}")
    
    # Categorize
    auto_labeled = []  # DLT headers or known contacts -> auto ham
    needs_review = []  # Unknown phone numbers -> manual review
    
    # Track phone number frequency (repeat contacts = likely real)
    phone_counts = Counter(m.get('address', '') for m in received if is_phone_number(m.get('address', '')))
    
    # Stats for contacts
    contact_matches = 0
    
    for msg in received:
        addr = msg.get('address', '')
        text = msg.get('text', '')
        
        if is_dlt_header(addr):
            # DLT header = registered business = ham
            auto_labeled.append({
                'text': text,
                'address': addr,
                'label': 0,  # ham
                'auto_reason': 'dlt_header'
            })
        elif is_phone_number(addr):
            # Check if it's a known contact
            contact_name = is_known_contact(addr, contacts)
            if contact_name:
                # Known contact = ham
                contact_matches += 1
                auto_labeled.append({
                    'text': text,
                    'address': addr,
                    'label': 0,
                    'auto_reason': f'contact:{contact_name}'
                })
            else:
                # Unknown phone number = needs review
                contact_freq = phone_counts.get(addr, 1)
                needs_review.append({
                    'text': text,
                    'address': addr,
                    'label': None,  # YOU FILL THIS: 0=ham, 1=scam
                    'contact_frequency': contact_freq,
                    'hint': 'frequent_contact' if contact_freq >= 3 else 'unknown'
                })
        else:
            # Unknown format - treat as needs review
            needs_review.append({
                'text': text,
                'address': addr,
                'label': None,
                'contact_frequency': 0,
                'hint': 'unknown_format'
            })
    
    # Stats
    dlt_count = sum(1 for m in auto_labeled if m.get('auto_reason') == 'dlt_header')
    print(f"\n{'='*60}")
    print("AUTO-LABELING RESULTS")
    print(f"{'='*60}")
    print(f"Auto-labeled as HAM:")
    print(f"  - DLT headers (businesses):  {dlt_count}")
    print(f"  - Known contacts:            {contact_matches}")
    print(f"  - Total auto-labeled:        {len(auto_labeled)}")
    print(f"\nNeeds manual review:           {len(needs_review)}")
    
    # Break down review list
    frequent = sum(1 for m in needs_review if m.get('contact_frequency', 0) >= 3)
    unknown = sum(1 for m in needs_review if m.get('contact_frequency', 0) < 3)
    print(f"  - Frequent senders (3+ msgs): {frequent}")
    print(f"  - Unknown/rare senders:       {unknown}")
    
    # Save auto-labeled (ready for training)
    output_data = {
        'auto_labeled_count': len(auto_labeled),
        'needs_review_count': len(needs_review),
        'messages': auto_labeled
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved auto-labeled messages to: {OUTPUT_FILE}")
    
    # Save review list (you edit this)
    # Sort by frequency (frequent contacts first = easier to mark as ham)
    needs_review.sort(key=lambda x: (-x.get('contact_frequency', 0), x.get('address', '')))
    
    review_data = {
        'instructions': 'Set label to 0 for ham (safe), 1 for scam. Frequent contacts are likely ham.',
        'total': len(needs_review),
        'messages': needs_review
    }
    with open(REVIEW_FILE, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)
    print(f"Saved messages needing review to: {REVIEW_FILE}")
    
    # Show sample of what needs review
    print(f"\n{'='*60}")
    print("SAMPLE MESSAGES FOR REVIEW (first 10)")
    print(f"{'='*60}")
    for i, msg in enumerate(needs_review[:10]):
        addr = msg['address']
        freq = msg['contact_frequency']
        hint = msg['hint']
        text = msg['text'][:80].replace('\n', ' ')
        print(f"\n{i+1}. [{addr}] (freq={freq}, {hint})")
        print(f"   {text}...")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"1. Open {REVIEW_FILE}")
    print(f"2. For each message, set 'label' to 0 (ham) or 1 (scam)")
    print(f"   - Frequent contacts (freq >= 3) are almost certainly ham")
    print(f"   - One-off phone numbers with links/offers = likely scam")
    print(f"3. Run: python merge_labels.py  (to combine into final training data)")

if __name__ == '__main__':
    main()
