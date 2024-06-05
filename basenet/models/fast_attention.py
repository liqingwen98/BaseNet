import torch
import torch.nn as nn

class MultiHeadFastSelfAttention(nn.Module):
    def __init__(self,                
                 num_attention_heads = 8,
                 input_dim = 512):
        super(MultiHeadFastSelfAttention, self).__init__()
        self.attention_head_size = input_dim//num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim = input_dim
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        if len(attention_mask.shape) == 4:
            query_for_score = query_for_score.unsqueeze(2)
            query_for_score = query_for_score.repeat(1, 1, seq_len, 1)    
            query_for_score += attention_mask
            query_weight = self.softmax(query_for_score)
        else:
            # add attention mask
            query_for_score += attention_mask
    
            # batch_size, num_head, 1, seq_len
            query_weight = self.softmax(query_for_score).unsqueeze(2)
    
        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).reshape(batch_size,-1,self.num_attention_heads*self.attention_head_size)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        if len(attention_mask.shape) == 4:
            query_key_score = query_key_score.unsqueeze(2)
            query_key_score = query_key_score.repeat(1, 1, seq_len, 1)
            query_key_score +=attention_mask
            query_key_weight = self.softmax(query_key_score)
        else:  
            # add attention mask             
            query_key_score +=attention_mask

            # batch_size, num_head, 1, seq_len
            query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)
        
        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value