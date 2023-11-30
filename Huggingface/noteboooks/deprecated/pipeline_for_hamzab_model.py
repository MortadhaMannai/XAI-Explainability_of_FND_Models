import transformers


class FakeNewsPipelineForHamzaB(transformers.TextClassificationPipeline):
    """
    custom pipeline for the model TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def postprocess(self, model_outputs, function_to_apply=None, return_all_scores=True):
        assert function_to_apply is None, 'If you want to use another function, ' \
                                          'please use TextClassificationPipeline instead.'
        outputs = model_outputs["logits"][0].numpy()
        scores = transformers.pipelines.text_classification.softmax(outputs)

        if self.model.config.label2id is None:
            # there is no label2id in the model config so we create it
            label2id = {}
            for id, label in self.model.config.id2label.items():
                label2id[label] = id
            self.model.config.label2id = label2id
            assert self.model.config.label2id is not None, 'Updating label2id in model config failed.'
        if return_all_scores:
            return [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in
                    enumerate(scores)]
        else:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}
