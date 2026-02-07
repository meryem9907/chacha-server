import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

class VisualLanguageModelForCharts():
    """
    Class for inference of a Visual Language Model from Hugging Face (e.g. OpenGVLab/InternVL3_5-8B-HF)
    """
    def load_model(self, model_path:str, force_cpu: bool):
        """
        Load vlm specified by the name in model card.

        Args:
            model_path (str): Path of the model specified in the model card in hugging face hub.
            force_cpu (bool): Select cpu specifically.
        """
        self.device = self.__pick_device(force_cpu)
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16 # because cpu has more gb in the server

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        torch_dtype=dtype,  
        device_map=None,     
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def run_vlm(self, prompt: str, dynamic_prompt:str, chart: Image.Image,  max_new_tokens: int=128) -> str:
        """
        Run inference for a prompt-chart pair.

        Args:
            prompt (str): Question on the chart.
            dynamic_prompt (str): Chain of thought provoking prompt for the system prompt.
            chart (PIL.Image.Image): Chart image.

        Returns:
            str: Response of the model.
        """
        img = chart.convert("RGB")

        messages = [
                {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an assistant that describes images for blind users. "
                        "Your responses must be short, spoken-friendly sentences."
                        "Do not use bullet points, lists, quotes, or special characters. "
                        "Speak naturally, as if reading aloud." + 
                        dynamic_prompt
                    ),
                }
            ],
        },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Applies a Jinja template to the messages and tokenizes it 
        inputs = self.processor.apply_chat_template(
            conversation = messages,
            add_generation_query=True,
            tokenize=True,
            return_tensors="pt", # return pytorch tensor (torch.Tesor(shape[batch, seq_len]))
            return_dict=True, # keys: input_ids, attention_mask
        )

        # make inputs device specific
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # generate encoded response
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # decode
        query_len = inputs["input_ids"].shape[1]
        generated_answer_ids = output_ids[:, query_len:]
        text = self.processor.batch_decode(generated_answer_ids, skip_special_tokens=True)[0]
        tts_friendly_resp = self.__tts_cleanup(text.strip())
        return tts_friendly_resp
    
    def __pick_device(self, force_cpu: bool) -> torch.device:
        """
        Pick cpu or a cuda supporting device if available.

        Args:
            force_cpu (bool): Select cpu specifically.

        Returns:
            torch.device: Device type
        """
        if force_cpu:
            return torch.device("cpu")
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
       
    def __tts_cleanup(self, text: str) -> str:
        """
        Remove special chars or hard-to-spell words for tts speech.

        Args:
            text (str): Text to cleanup.

        Returns:
            str: Cleaned-up text.
        """
        replacements = {
            "approximately": "about",
            "Approximate": "About",
            "\"": "",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = text.replace("\n\n", ". ").replace("\n", " ")
        return text.strip()