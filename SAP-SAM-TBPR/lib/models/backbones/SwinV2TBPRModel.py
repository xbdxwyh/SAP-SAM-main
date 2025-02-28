import torch

from typing import Optional, Tuple, Union

from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2Encoder,
    Swinv2EncoderOutput
)

from transformers import Swinv2Model

class Swinv2TBPREncoder(Swinv2Encoder):
#class Swinv2TBPREncoder(nn.Module):
    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__(config, grid_size, pretrained_window_sizes)

        # Copied from transformers.models.swin.modeling_swin.SwinEncoder.forward with SwinEncoderOutput->Swinv2EncoderOutput
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        multi_stage_feature_list = []
        for i, layer_module in enumerate(self.layers):
            #print(i)
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states, input_dimensions, layer_head_mask
                )
            else:
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            multi_stage_feature_list.append(hidden_states)
            output_dimensions = layer_outputs[1]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)

            if output_hidden_states:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=multi_stage_feature_list,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class SwinV2TBPRModel(Swinv2Model):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config, add_pooling_layer, use_mask_token)
        self.encoder = Swinv2TBPREncoder(config, self.embeddings.patch_grid)