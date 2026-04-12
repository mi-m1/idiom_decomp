
from openai import OpenAI

import os

class OpenAIPrompter:
    def __init__(self, key, model,):
        """
        Initialize the OpenAIPrompter class.

        Parameters:
        - api_key (str): Your OpenAI API key.
        - model (str): The model to use for the prompts. Default is "gpt-3.5-turbo".
        """
        self.key = key
        self.model = model

    # def prompt(self, prompt_text, max_tokens, temperature, top_p, n):
    def prompt(self, prompt_text):
        """
        Send a prompt to the OpenAI model and return the response.

        Parameters:
        - prompt_text (str): The text prompt to send to the model.
        - max_tokens (int): The maximum number of tokens to generate in the response.
        - temperature (float): Sampling temperature to use. Higher values means the model will take more risks.
        - top_p (float): Nucleus sampling parameter. The model will consider the smallest set of tokens with cumulative probability top_p.
        - n (int): Number of completions to generate for the prompt.

        Returns:
        - response (str): The generated response from the model.
        """

        # client = OpenAI(self.api_key)
        client=OpenAI(api_key = self.key)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                              "type": "text",
                              "text": prompt_text,
                            }
                        ]

                    }
                ]
                # max_tokens=max_tokens,
                # temperature=temperature,
                # top_p=top_p,
                # n=n
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"


def run_extraction(sentence, exact_form, model):
    prompter = OpenAIPrompter(
        key=os.environ["OPENAI_API_KEY"], 
        model=model,
    )

    # prompt_text = f"Please extract the idiom from the sentence. For example, Sentence: 'How have you weathered the storm? Output: 'weathered the storm'. Only output the idiom as how it occurs in the Sentence, do not say anything else. \nSentence:{sentence}, output:"
    prompt_text = f"Given a sentence, and how an idiom appears in the sentence, please extract the base form of the idiom. For example, Sentence: 'And then , the Daily Telegraph discovered ‘ the truth’ : ‘ privatisation of Britain 's water industry … runs against the grain of Mr Delors’ social strategy.' Idiom in sentence: 'runs against the grain' Base form of idiom: 'against the grain'. Only output the base form of the idiom as it would appear in isolation, do not say anything else. \nSentence:{sentence}, Exact form in sentence: {exact_form} output:"

    response = prompter.prompt(prompt_text)

    print("Response from OpenAI model:")
    print(response)

    return response

def run_impli_pipeline(model, path_to_dataset, save=True):

    
    filename = os.path.basename(path_to_dataset)
    filename = filename.replace(".tsv", ".csv")

    if "impli30" in filename:
        impli = pd.read_csv(path_to_dataset, sep="\t",)
    else:
        # impli = pd.read_csv(path_to_dataset, sep="\t", header=None, names=["premise", "hypothesis"])
        impli = pd.read_csv(path_to_dataset,)

    impli["base_form"] = impli.progress_apply(lambda row: run_extraction(row["premise"], row["extracted_idiom"], model), axis=1)

    if save == True:
        impli.to_csv(f"data/processed/baseFrom_{model}_{filename}", index=False)

    print(f"Done and saved to data_processed/{model}_{filename}")


if __name__ == "__main__":

    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas()

    # #impli30 pipeline
    # run_impli_pipeline(
    #     model="gpt-5.1-2025-11-13",
    #     path_to_dataset="data/impli30.tsv"
    # )

    # run_impli_pipeline(
    #     model="gpt-5-nano-2025-08-07",
    #     path_to_dataset="data/impli30.tsv"
    # )

    #impli pipeline
    run_impli_pipeline(
        model="gpt-5.1-2025-11-13",
        path_to_dataset="data/processed/gpt-5.1-2025-11-13_manual_e.csv",
        save=True,
    )