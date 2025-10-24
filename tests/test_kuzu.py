from mem0 import Memory
"""
config = {
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "vllm_base_url": "http://localhost:8000/v1",
        },
    },
}
"""
config = {
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db": "/tmp/mem0-g-test.kuzu",
        },
    },
    "vector_store": {
        "provider": "kuzu",
        "config": {
            "db": "/tmp/mem0-v-test.kuzu",
            "embedding_dims": 1536,
        },
    },
}
m = Memory.from_config(config_dict=config)

"""{'results': [{'id': 'e2a5114e-3c2e-4646-9449-002348e40cc0', 'memory': 'Likes pizza', 'event': 'ADD'}], 'relations': {'deleted_entities': [[], []], 'added_entities': [[{'source': 'user_id:_alice', 'relationship': 'likes', 'target': 'pizza'}]]}}"""
r = m.add("I like pizza", user_id="alice")
print(r)

"""{'results': [{'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'event': 'ADD'}], 'relations': {'deleted_entities': [[], []], 'added_entities': [[{'source': 'user_id:_alice,_agent_id:_food-assistant', 'relationship': 'likes', 'target': 'pizza'}]]}}"""
r = m.add("I like pizza", user_id="alice", agent_id="food-assistant")
print(r)

"""{'results': [], 'relations': {'deleted_entities': [], 'added_entities': [[{'source': 'user_id:_alice,_agent_id:_food-assistant,_run_id:_session-123', 'relationship': 'likes', 'target': 'pizza'}]]}}"""
r = m.add("I like pizza", user_id="alice", agent_id="food-assistant", run_id="session-123")
print(r)

"""{'results': [{'id': 'e2a5114e-3c2e-4646-9449-002348e40cc0', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'created_at': '2025-10-23T23:36:32.348678-07:00', 'updated_at': None, 'user_id': 'alice', 'agent_id': None}, {'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': [{'source': 'user_id:_alice', 'relationship': 'likes', 'target': 'pizza'}, {'source': 'user_id:_alice,_agent_id:_food-assistant', 'relationship': 'likes', 'target': 'pizza'}, {'source': 'user_id:_alice,_agent_id:_food-assistant,_run_id:_session-123', 'relationship': 'likes', 'target': 'pizza'}]}"""
r = m.get_all(user_id="alice")
print(r)

"""{'results': [{'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': [{'source': 'user_id:_alice,_agent_id:_food-assistant', 'relationship': 'likes', 'target': 'pizza'}, {'source': 'user_id:_alice,_agent_id:_food-assistant,_run_id:_session-123', 'relationship': 'likes', 'target': 'pizza'}]}"""
r = m.get_all(user_id="alice", agent_id="food-assistant")
print(r)

"""{'results': [{'id': 'e2a5114e-3c2e-4646-9449-002348e40cc0', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'created_at': '2025-10-23T23:36:32.348678-07:00', 'updated_at': None, 'user_id': 'alice', 'agent_id': None}, {'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': [{'source': 'user_id:_alice,_agent_id:_food-assistant,_run_id:_session-123', 'relationship': 'likes', 'target': 'pizza'}]}"""
r = m.get_all(user_id="alice", run_id="session-123")
print(r)

"""{'results': [{'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': [{'source': 'user_id:_alice,_agent_id:_food-assistant,_run_id:_session-123', 'relationship': 'likes', 'target': 'pizza'}]}"""
r = m.get_all(user_id="alice", agent_id="food-assistant", run_id="session-123")
print(r)

"""{'results': [{'id': 'e2a5114e-3c2e-4646-9449-002348e40cc0', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'score': 0.1682094697343749, 'created_at': '2025-10-23T23:36:32.348678-07:00', 'updated_at': None, 'user_id': 'alice', 'agent_id': None}, {'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'score': 0.1682094697343749, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': []}"""
r = m.search("tell me my name.", user_id="alice")
print(r)

"""{'results': [{'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'score': 0.1682094697343749, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': []}"""
r = m.search("tell me my name.", user_id="alice", agent_id="food-assistant")
print(r)

"""{'results': [{'id': 'e2a5114e-3c2e-4646-9449-002348e40cc0', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'score': 0.1682094697343749, 'created_at': '2025-10-23T23:36:32.348678-07:00', 'updated_at': None, 'user_id': 'alice', 'agent_id': None}, {'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'score': 0.1682094697343749, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': []}"""
r = m.search("tell me my name.", user_id="alice", run_id="session-123")
print(r)

"""{'results': [{'id': '6f80dcc9-85f4-47e8-8cd3-24f8b12ba510', 'memory': 'Likes pizza', 'hash': '92128989705eef03ce31c462e198b47d', 'metadata': {'app_id': None, 'labels': []}, 'score': 0.1682094697343749, 'created_at': '2025-10-23T23:36:34.797068-07:00', 'updated_at': '2025-10-23T23:36:37.283104-07:00', 'user_id': 'alice', 'agent_id': 'food-assistant', 'run_id': 'session-123'}], 'relations': []}"""
r = m.search("tell me my name.", user_id="alice", agent_id="food-assistant", run_id="session-123")
print(r)

"""{'message': 'Memories deleted successfully!'}"""
r = m.delete_all(user_id="alice")
print(r)

"""{'message': 'Memories deleted successfully!'}"""
r = m.delete_all(user_id="alice", agent_id="food-assistant")
print(r)

"""{'message': 'Memories deleted successfully!'}"""
r = m.delete_all(user_id="alice", run_id="session-123")
print(r)

"""{'message': 'Memories deleted successfully!'}"""
r = m.delete_all(user_id="alice", agent_id="food-assistant", run_id="session-123")
print(r)
