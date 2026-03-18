#RFLPA Server - Round 1
#Collect gradient shares from clients.
#NOTE: Encryption removed - shares are passed directly.

from __future__ import annotations

from typing import Dict, Optional

from fl.utils_server import ServerState

print("[DEBUG][round1_server.py] Module loaded")


class Round1Handler:
    
    #Collect gradient shares.
    
    #Message from client contains:
    #    - shares: list of ShareMsg (sender_id, recipient_id, block_idx, alpha, share_val)
    
    
    def __init__(self, state: ServerState):
        print("[DEBUG][round1_server.py] Round1Handler initialized")
        self.state = state
    
    def receive_message(self, client_id: int, message: Dict) -> bool:
        
        #Receive Round 1 message from client.
        
        
        #client_id: The sending client's ID
        #message: Dict containing shares (or legacy format with commitment/encrypted_shares)
            
        
        #return True if message was accepted, False otherwise
        
        print(f"[DEBUG][round1_server.py] receive_message called: client_id={client_id}")
        
        if client_id not in self.state.U0:
            print(f"[DEBUG][round1_server.py] client_id={client_id} not in U0, rejecting")
            return False
        
        # Accept new simplified format (just shares) or legacy format
        if 'shares' in message:
            # New simplified format - no encryption
            print(f"[DEBUG][round1_server.py] client_id={client_id} accepted (simplified format), shares count={len(message['shares'])}")
            self.state.round1_messages[client_id] = message
            self.state.U1.add(client_id)
            return True
        
        # Legacy format with encryption (for backwards compatibility)
        required = ['commitment', 'encrypted_shares', 'signatures']
        if all(k in message for k in required):
            print(f"[DEBUG][round1_server.py] client_id={client_id} accepted (legacy format)")
            self.state.round1_messages[client_id] = message
            self.state.U1.add(client_id)
            return True
        
        print(f"[DEBUG][round1_server.py] client_id={client_id} rejected (invalid message format)")
        return False
    
    def is_complete(self) -> bool:
        #at least K clients responded in Round 1.
        complete = len(self.state.U1) >= self.state.min_clients
        print(f"[DEBUG][round1_server.py] is_complete: {complete} (U1={len(self.state.U1)}, min={self.state.min_clients})")
        return complete
    
    def prepare_response(self, target_client: int) -> Optional[Dict]:
        
        #Prepare Round 1 response for client j.
        #Returns shares intended for this client.
        
        
        #target_client: The client to prepare response for
        #return Response dict or None if client not in U1
        
        print(f"[DEBUG][round1_server.py] prepare_response called: target_client={target_client}")
        
        if target_client not in self.state.U1:
            print(f"[DEBUG][round1_server.py] target_client={target_client} not in U1")
            return None
        
        response = {
            'shares': {},
        }
        
        for sender_id in self.state.U1:
            if sender_id == target_client:
                continue
            
            msg = self.state.round1_messages[sender_id]
            
            # New simplified format
            if 'shares' in msg:
                response['shares'][sender_id] = msg['shares']
            # Legacy format
            elif 'encrypted_shares' in msg:
                if target_client in msg['encrypted_shares']:
                    response['shares'][sender_id] = msg['encrypted_shares'][target_client]
        
        print(f"[DEBUG][round1_server.py] prepare_response completed: {len(response['shares'])} senders")
        return response
    
    def get_respondent_count(self) -> int:
        #Return number of clients that responded in Round 1.
        count = len(self.state.U1)
        print(f"[DEBUG][round1_server.py] get_respondent_count: {count}")
        return count
    
    def get_respondents(self) -> set:
        #Return set of clients that responded in Round 1.
        print(f"[DEBUG][round1_server.py] get_respondents: {self.state.U1}")
        return self.state.U1.copy()
