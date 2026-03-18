#RFLPA Server - Round 2
#Collect partial norm/cosine similarity shares from clients.
#NOTE: Encryption removed - shares are passed directly.

from __future__ import annotations

from typing import Dict, Optional

from fl.utils_server import ServerState

print("[DEBUG][round2_server.py] Module loaded")


class Round2Handler:
    
    #Round 2: Collect partial norm/cosine similarity shares.
    
    #Message from client contains:
    #    - cs_shares: list of ShareMsg for cosine similarity
    #    - nr_shares: list of ShareMsg for norm
    
    def __init__(self, state: ServerState):
        print("[DEBUG][round2_server.py] Round2Handler initialized")
        self.state = state
    
    def receive_message(self, client_id: int, message: Dict) -> bool:
        
        #Receive Round 2 message from client.
        
        print(f"[DEBUG][round2_server.py] receive_message called: client_id={client_id}")
        
        if client_id not in self.state.U1:
            print(f"[DEBUG][round2_server.py] client_id={client_id} not in U1, rejecting")
            return False
        
        # Accept new simplified format (direct shares) or legacy format
        if 'cs_shares' in message and 'nr_shares' in message:
            # New simplified format - no encryption
            print(f"[DEBUG][round2_server.py] client_id={client_id} accepted, cs={len(message['cs_shares'])}, nr={len(message['nr_shares'])}")
            self.state.round2_messages[client_id] = message
            self.state.U2.add(client_id)
            return True
        
        # Legacy format with encryption (for backwards compatibility)
        required = ['commitment', 'encrypted_shares', 'signatures']
        if all(k in message for k in required):
            print(f"[DEBUG][round2_server.py] client_id={client_id} accepted (legacy format)")
            self.state.round2_messages[client_id] = message
            self.state.U2.add(client_id)
            return True
        
        print(f"[DEBUG][round2_server.py] client_id={client_id} rejected (invalid message format)")
        return False
    
    def is_complete(self) -> bool:
        #Check if at least K clients responded in Round 2.
        complete = len(self.state.U2) >= self.state.min_clients
        print(f"[DEBUG][round2_server.py] is_complete: {complete} (U2={len(self.state.U2)}, min={self.state.min_clients})")
        return complete
    
    def prepare_response(self, target_client: int) -> Optional[Dict]:
        
        #Prepare Round 2 response for client j.
        #Returns cs/nr shares intended for this client.
        
        print(f"[DEBUG][round2_server.py] prepare_response called: target_client={target_client}")
        
        if target_client not in self.state.U2:
            print(f"[DEBUG][round2_server.py] target_client={target_client} not in U2")
            return None
        
        response = {
            'cs_shares': {},
            'nr_shares': {},
        }
        
        for sender_id in self.state.U2:
            if sender_id == target_client:
                continue
            
            msg = self.state.round2_messages[sender_id]
            
            # New simplified format
            if 'cs_shares' in msg:
                response['cs_shares'][sender_id] = msg['cs_shares']
                response['nr_shares'][sender_id] = msg['nr_shares']
            # Legacy format
            elif 'encrypted_shares' in msg:
                if target_client in msg['encrypted_shares']:
                    response['cs_shares'][sender_id] = msg['encrypted_shares'][target_client]
        
        print(f"[DEBUG][round2_server.py] prepare_response completed: {len(response['cs_shares'])} senders")
        return response
    
    def get_respondent_count(self) -> int:
        #Return number of clients that responded in Round 2.
        count = len(self.state.U2)
        print(f"[DEBUG][round2_server.py] get_respondent_count: {count}")
        return count
    
    def get_respondents(self) -> set:
        #Return set of clients that responded in Round 2.
        print(f"[DEBUG][round2_server.py] get_respondents: {self.state.U2}")
        return self.state.U2.copy()
