# from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
#                           TextClassificationPipeline)

# bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
# bert_tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)
# model = (AutoModelForSequenceClassification
#             .from_pretrained(bert_ckpt, torchscript=True).to("cpu"))
# model.eval()

# input_names = ['input_ids', 'attention_mask', 'token_type_ids']
# input_shape = [(1, 7), (1, 7), (1, 7)]
# dummy_input = [torch.randint(1, x) for x in input_shape]


# # Creating the trace
# traced_model = torch.jit.trace(model, dummy_input)
# torch.jit.save(traced_model, "bert.pt")

import torch
from transformers.pipelines import Pipeline
from transformers.convert_graph_to_onnx import infer_shapes, ensure_valid_input


def transformers_to_pt(nlp: Pipeline, export_file_path: str):

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
        ordered_input_names, model_args = ensure_valid_input(nlp.model, tokens, input_names)

        traced_model = torch.jit.trace(nlp.model, model_args, strict=False)
        torch.jit.save(traced_model, export_file_path)


if __name__ == '__main__':
    from transformers import pipeline, BertTokenizer, BertModel

    pipe = pipeline('fill-mask', model = 'bert-base-uncased')
    output = pipe("Hello I'm a [MASK] model.")
    print(output)

    transformers_to_pt(pipe, 'env/test.pt')
    model = torch.load('env/test.pt')

    