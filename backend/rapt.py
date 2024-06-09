import dspy

class OracleSignature(dspy.Signature):
    """You will be given a question and context. You need to answer the question with explanation based on the context given. If the answer doesn't lie in the context, say I don't know"""
    question = dspy.InputField(desc="Question asked")
    context = dspy.InputField(desc="Potentially related passages")
    answer = dspy.OutputField(desc="Answer to the question based on the given context, just give answer, and nothing else")

class DistractorSignature(dspy.Signature):
    """You will be given a question and context. You need to answer the question with explanation based on the context given. If the answer doesn't lie in the context, say I don't know"""
    question = dspy.InputField(desc="Question asked")
    context = dspy.InputField(desc="Potentially unrelated passages")
    answer = dspy.OutputField(desc="Give 2-3 lines reason you cannot answer the given question as the context is insufficient or unrelated")

class OracleRAFT(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate_answer = dspy.ChainOfThought(OracleSignature)

    def forward(self, question):
        context = self.retrieve(question).toDict()["passages"]
        prediction = self.generate_answer(question=question, context=context)
        # return dspy.Prediction(context=context, answer=prediction.answer)
        return prediction
    
class DistractorRAFT(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=30)
        self.generate_answer = dspy.ChainOfThought(DistractorSignature)

    def forward(self, question):
        context = self.retrieve(question).toDict()["passages"][-5:]
        prediction = self.generate_answer(question=question ,context=context)
        # return dspy.Prediction(context=context, answer=prediction.answer)
        return prediction