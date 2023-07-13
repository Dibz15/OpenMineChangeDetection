from gimpfu import *
import time

def toggle_layer(image, drawable, layer, frequency, duration):
    # Calculate the end time
    end_time = time.time() + duration

    # While the current time is less than the end time
    while time.time() < end_time:
        # Toggle the layer's visibility
        layer.visible = not layer.visible

        # Redraw the image
        drawable.flush()
        gimp.displays_flush()

        # Wait for the specified frequency
        time.sleep(frequency)

register(
    "python_fu_toggle_layer",
    "Toggle Layer",
    "Continuously toggles the visibility of a specific layer at a certain frequency",
    "Your Name",
    "Your Name",
    "2023",
    "<Image>/Filters/Toggle Layer...",
    "*",
    [
        (PF_LAYER, "layer", "Layer to toggle", None),
        (PF_FLOAT, "frequency", "Toggle frequency (in seconds)", 0.5),
        (PF_INT, "duration", "Duration (in seconds)", 10)
    ],
    [],
    toggle_layer)

main()
