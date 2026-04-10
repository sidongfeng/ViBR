import base64
from openai import OpenAI

"""
Functions to interact with OpenAI GPT-4o for visual app state comparison, action region prediction,
and relevant region identification for Android GUI screenshots.

WARNING: Remove hardcoded API key before sharing or pushing to public repositories.
"""

# TODO: Remove API key before sharing code! Never hardcode secrets in production.
client = OpenAI(api_key="put-your-api-key-here")

def encode_image(image_path):
    """Read an image file and return its base64-encoded string (UTF-8)."""
    print(image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_gpt_state_consistency(start_img, live_img, action="", target_region=""):
    """
    Compares two Android screenshots to determine if their UI state is functionally equivalent,
    using GPT-4o via the OpenAI API.

    Args:
        start_img (str): Path to reference image.
        live_img (str): Path to live/current image.
        action (str): The action to check for (optional).
        target_region (str): Target UI region (optional).

    Returns:
        str: GPT-4o response in JSON: {"same_state": "yes"} or {"same_state": "no", ...}
    """
    b64_start = encode_image(start_img)
    b64_live = encode_image(live_img)

    prompt = (
        "You are given two screenshots of an Android interface:\n"
        "1. The first image is the REFERENCE state from a stable app video.\n"
        "2. The second image is the CURRENT real-time app state.\n"
        "\n"
        "You also get a possible action and region that has to be executed to reach the target state. Take this into account but also keep in mind that something else could be the action. \n"
        "Action: " + action + "\n Region: {target_region}" +
        "Your task is to determine if the current screen is functionally consistent with the reference.\n"
        "That means: Can the user perform the same action from the current screen as in the reference?\n"
        "\n"
        "- Minor differences in layout, text alignment, icon position or additional items that do not influence the action DO NOT matter.\n"
        "- For home screens or app drawers, the presence of extra app icons, widgets, or a different order of icons DOES NOT matter, as long as the same action can be performed from both screens.\n"
        "- Focus on whether the same buttons, inputs, or menus are present and usable. Sometimes the keyboard or something can block some parts, this still means the state is consistent. \n"
        "- Ignore small stylistic or timing variations (e.g., animation state, different time shown, small icon differences).\n"
        "- Cases like the home screen or similar, where icons can be ordered differently do not matter if the same action can be performed."
        "\n"
        "Respond strictly in the following JSON format:\n"
        "{ \"same_state\": \"yes\" } or { \"same_state\": \"no\", \"description\": \"<reason>\" }"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_start}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_live}"}},
            ]}
        ]
    )

    print("Consistency Response from GPT-4o:", response.choices[0].message.content)

    return response.choices[0].message.content.strip().lower()

def ask_gpt_for_action_region(start_img, stop_img, live_img, predicted_action, relevant_indices=None):
    """
    Uses GPT-4o to infer which action and UI region should be executed on the current (live) screen
    to reproduce a state transition observed in start/stop images.

    Args:
        start_img (str): Path to start image.
        stop_img (str): Path to stop image (after action).
        live_img (str): Path to live/current image.
        predicted_action (str): Action type (e.g., tap, swipe).
        relevant_indices (list): Optionally, region indices.

    Returns:
        str: JSON response from GPT-4o describing action and region.
    """
    b64_start = encode_image(start_img)
    b64_stop = encode_image(stop_img)
    b64_live = encode_image(live_img)

    # Prompt for action inference using start, stop, and current screenshots, based on region indices
    prompt_instruction_region = '''
    Your goal is to reproduce the action {predicted_action} from the GUI recording on a real device. I show you the three GUI screenshots by order. In the recording, the interaction with the highlighted purple region in the first GUI leads to the second GUI. The current GUI on your device is shown as the third GUI, on which element should you perform the action to achieve the same transition? Please follow the primitive in action space.
 
    ### Possible Actions: 
    1. **tap** - Taps a location on screen. - Example: { "action": "tap", "region": 2, "description": "Tap center of screen to open app." } 
 
    2. **swipe** - Swipes from one point to another. - Example: { "action": "swipe", "from": [540, 1600], "to": [540, 400], "duration": 500, "description": "Swipe up to scroll." } 
 
    3. If you see the keyboard on the GUI screen, it is highly possible is a **input_text** - Types text into a focused input field. - Example: { "action": "input_text", "text": "hello world", "description": "Type search query." }
 
    4. **back** - Presses Android back button. - Example: { "action": "back", "description": "Go back to previous screen." } 
 
    5. **home** - Goes to Android home screen. - Example: { "action": "home", "description": "Return to home." } 
 
    6. **wait** - Waits for a specified duration. - Example: { "action": "wait", "duration": 1500, "description": "Wait for animation to finish." } 
 
    7. **no action** - No action is needed. - Example: { "action": "no action", "description": "No Action needed." } 
 
    Return a **JSON object** describing the required action. Do not include any other text or explanation.
    '''

    response = client.chat.completions.create(
        model="gpt-4o",
        # temperature=0.2,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt_instruction_region },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_start}", "detail": "low"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_stop}", "detail": "low"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_live}", "detail": "high"}}
            ]}
        ]
    )

    print("Region Action Response from GPT-4o:", response.choices[0].message.content)

    return response.choices[0].message.content

def ask_gpt_for_relevant_regions(start_img_path, stop_img_path):
    """
    Sends start and stop images to GPT-4o and asks which UI regions are most relevant for
    the transition, and predicts the action type.

    Args:
        start_img_path (str): Path to start (reference) image.
        stop_img_path (str): Path to stop (after interaction) image.

    Returns:
        str: JSON response with relevant regions and predicted action.
    """
    prompt_instruction_relevant_regions = """
      You are given two screenshots of an Android interface:

      1. The first image is the REFERENCE state before an interaction.
      2. The second image is the FOLLOW-UP state after the interaction.

      You are also given a list of **interactive UI regions** detected in the reference image. Each region includes:
      - A numeric index
      - A bounding box
      - A phrase describing the region (e.g., "button", "text field")

      Your task is to determine which of these regions are **most likely involved** in the transition between the two states.

      - Focus on regions that, if interacted with, could explain the visual change between the first and second image.
      - Minor layout shifts or content changes are not enough — identify only regions that are plausible interaction targets.
      - Use the phrases and bounding boxes to reason about the intent of the user.
      - When pointers or animations on a button or similar can be seen prioritize the region around it.

      You must also predict the type of user action that caused the change. Choose only from the following actions:
      ["tap", "double_tap", "long_press", "swipe", "input_text", "back", "home", "wait", "no action"]

      Respond strictly in the following JSON format — do not include any other text or explanation. If no regions are relevant, return an empty list:
      { "target_regions": [int, int, ...], "predicted_action": "<action>" }
      """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_instruction_relevant_regions + "\n\nScreenshots are attached below."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(start_img_path)}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(stop_img_path)}", "detail": "high"}}
            ]
        }]
    )

    print("Relevant Region Response from GPT-4o:", response.choices[0].message.content)
    return response.choices[0].message.content
