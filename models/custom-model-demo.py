"""Define custom model based on BERT."""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertModel
from transformers.modeling_outputs import TokenClassifierOutput


class CustomBERTModel(nn.Module):
    """Custom model based on `BertForTokenClassification`.
        - Construct model from BertConfig.

    `BertForTokenClassification` adds droupout and linear layer
    on top of the hidden-states output from BERTT. Custom the 
    model by modifyng theses addtional layers.

    Fine-tuning BERT requires 4x more runtime. We freeze it. 
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, self.num_labels)

        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

              
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # BERT Output
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output: (Batch_size, max_seq_len, dims)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(loss=loss,
                                     logits=logits,
                                     hidden_states=outputs.hidden_states, # From BERT
                                     attentions=outputs.attentions)

def model():
    return CustomBERTModel
