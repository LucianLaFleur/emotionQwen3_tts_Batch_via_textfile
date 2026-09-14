Adds emotion control, 1 generation per line. 
See the text file for expected formatting:

text to be spoken |emotion1|emotion2|emotion3|0.0-2.0 intensity
emotion2 and 3 can be set to None, but emotion1 is always needed (neutral works for no emotion in particular)

1) No input text from comfy/UI
2) control text to be spoken, 3 emotion controls, and intensity from an input textfule
3) No output audio node needed (though output, audio preview allows the output to generate), though naming is managed automatically based off content of the lines.

This extends the basic file made by Dawizzer.
https://github.com/Dawizzer/ComfyUI-Qwen3TTS-Emotional
