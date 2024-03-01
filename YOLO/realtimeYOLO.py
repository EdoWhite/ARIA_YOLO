import argparse
import numpy as np
import sys
import cv2
import time
import os
import subprocess
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device_ip", help="IP address to connect to the device over wifi"
    )

    parser.add_argument(
    "--yolo_weights", 
    required=True,
    type=str
    )

    parser.add_argument(
    "--yolo_cfg", 
    required=True,
    type=str
    )

    return parser.parse_args()

class StreamingClientObserver():
    def __init__(self):
        self.images = {}

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")

def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)

# Load and prepare YOLO model
def load_yolo(args):
    net = cv2.dnn.readNet(args.yolo_weights, args.yolo_cfg)
    classes = []
    with open("./yolo_models/coco.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if unconnected_out_layers.ndim == 1:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    return net, classes, output_layers

# Detect objects using YOLO
def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs

def draw_labels_and_boxes(img, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 2)
    return img

# Process the detections
def process_detections(outputs, width, height):
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def device_stream(args):
    if sys.platform.startswith("linux"):
        update_iptables()
    # Set debug level
    aria.set_log_level(aria.Level.Info)
    # Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    # Connect to the device
    device = device_client.connect()
    # Retrieve the streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client
    # Set custom config for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    # Streaming type
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    # Use ephemeral streaming certificates
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    # Start streaming
    streaming_manager.start_streaming()
    # Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")
    return streaming_manager, streaming_client, device_client, device


def device_subscribe(streaming_client):
    # Configure subscription
    config = streaming_client.subscription_config
    config.subscriber_data_type = (aria.StreamingDataType.Rgb)
    # Take most recent frame
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config
    # Set the observer
    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    # Start listening
    print("Start listening to image data")
    streaming_client.subscribe()
    return observer

# Main function
def main():
    args = parse_args()
    streaming_manager, streaming_client, device_client, device = device_stream(args)
    observer = device_subscribe(streaming_client)

    rgb_window = "Aria RGB"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1080, 1080)
    #cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)
    net, classes, output_layers = load_yolo(args)

    while not quit_keypress():
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape

            # YOLO object detection
            outputs = detect_objects(rgb_image, net, output_layers)
            boxes, confidences, class_ids = process_detections(outputs, width, height)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            img_with_boxes = draw_labels_and_boxes(rgb_image.copy(), boxes, confidences, class_ids, classes)
            cv2.imshow(rgb_window, img_with_boxes)
            del observer.images[aria.CameraId.Rgb]

    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
        
main()
