# Terminator Vision Simulation

![Terminator Vision](./assets/terminator_vision_banner.png)

## Overview

This project simulates the iconic "Terminator Vision" from the classic movie *The Terminator*. The simulation replicates the red-tinted HUD (Heads-Up Display) seen through the eyes of the Terminator, featuring data analysis, object detection, and threat assessment.

## Features

- **Red-Tinted HUD**: A distinctive red overlay that mimics the visual style of the Terminator's vision.
- **Object Detection**: Identifies and highlights objects in real-time using bounding boxes and labels.
- **Data Analysis**: Displays various data points, such as object speed, distance, and threat level.
- **Threat Assessment**: Classifies objects based on threat level, with different color codes.
- **Customizable Interface**: Modify the layout, colors, and data points to fit different use cases.

## Screenshots

![HUD Screenshot](./assets/hud_screenshot.png)
![Threat Detection](./assets/threat_detection.png)

## Installation

To get started with the simulation, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/terminator-vision-simulation.git
    cd terminator-vision-simulation
    ```

2. **Install Dependencies**:
    Make sure you have Python 3.8+ installed. Then, run:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Simulation**:
    ```bash
    python main.py
    ```

## Usage

Once the simulation is running, you can:

- Move the camera feed to simulate different perspectives.
- Activate object detection by pressing `D`.
- Adjust HUD parameters in the `config.json` file.
- Use the `ESC` key to exit the simulation.

## Configuration

You can customize the simulation by editing the `config.json` file:

- **hud_color**: Set the color of the HUD overlay (default is red).
- **threat_levels**: Define thresholds for low, medium, and high threat levels.
- **object_detection**: Toggle object detection on/off.

Example `config.json`:
```json
{
    "hud_color": "#FF0000",
    "threat_levels": {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8
    },
    "object_detection": true
}
```

## Contributing

We welcome contributions to enhance this project! Please submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- Inspired by *The Terminator* movie, directed by James Cameron.
- Special thanks to the open-source community for providing the tools and libraries used in this project.

---