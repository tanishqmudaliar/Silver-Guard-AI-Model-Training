"""
Merge auto-labeled and manually reviewed messages into final training data.
Run this AFTER you've labeled messages in review.json
"""

import json
from pathlib import Path

OUTPUT_FILE = "output.json"      # auto-labeled messages
REVIEW_FILE = "review.json"      # manually reviewed messages
FINAL_FILE = "training_data.json"  # combined output for retraining

def main():
    # Load auto-labeled
    if not Path(OUTPUT_FILE).exists():
        print(f"ERROR: {OUTPUT_FILE} not found. Run label_sms.py first.")
        return
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        auto_data = json.load(f)
    auto_msgs = auto_data.get('messages', [])
    
    # Load reviewed
    if not Path(REVIEW_FILE).exists():
        print(f"ERROR: {REVIEW_FILE} not found. Run label_sms.py first.")
        return
    
    with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
        review_data = json.load(f)
    review_msgs = review_data.get('messages', [])
    
    # Check for unlabeled messages
    unlabeled = [m for m in review_msgs if m.get('label') is None]
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} messages in {REVIEW_FILE} still have label=null")
        print("These will be SKIPPED. Edit review.json and set labels first.")
        print(f"\nFirst unlabeled: {unlabeled[0].get('address')} - {unlabeled[0].get('text', '')[:50]}...")
    
    # Filter to only labeled messages
    labeled_review = [m for m in review_msgs if m.get('label') is not None]
    
    # Combine
    all_msgs = []
    
    # Add auto-labeled (already have label=0)
    for m in auto_msgs:
        all_msgs.append({
            'text': m['text'],
            'address': m['address'],
            'label': 0
        })
    
    # Add manually reviewed
    for m in labeled_review:
        all_msgs.append({
            'text': m['text'],
            'address': m['address'],
            'label': m['label']
        })
    
    # Stats
    ham = sum(1 for m in all_msgs if m['label'] == 0)
    scam = sum(1 for m in all_msgs if m['label'] == 1)
    
    print(f"\n{'='*60}")
    print("FINAL TRAINING DATA")
    print(f"{'='*60}")
    print(f"Total messages: {len(all_msgs)}")
    print(f"Ham (safe):     {ham}")
    print(f"Scam:           {scam}")
    print(f"Skipped (unlabeled): {len(unlabeled)}")
    
    # Save
    final_data = {
        'total': len(all_msgs),
        'ham_count': ham,
        'scam_count': scam,
        'messages': all_msgs
    }
    with open(FINAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to: {FINAL_FILE}")
    print(f"\nUpload {FINAL_FILE} to Colab and use it in Cell 9 for retraining.")

if __name__ == '__main__':
    main()
