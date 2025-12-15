import json
from pathlib import Path

records = [
    {
        "id": f"rec_{i:04d}",
        # "payload": {
        #     "message": f"test message {i}",
        #     "index": i,
        #     "data": {"value": f"data_{i}"}
        # }
        "payload": {
            "name": "Apple MacBook Pro 16",
            "data": {
                "year": 2019,
                "price": 1849.99,
                "CPU model": "Intel Core i9",
                "Hard disk size": "1 TB"
            }
        }
    }
    for i in range(5000)
]

output_path = Path(__file__).parent / "test_dataset_large.json"
with open(output_path, 'w') as f:
    json.dump(records, f, indent=2)

print(f"Generated {len(records)} records in {output_path}")
