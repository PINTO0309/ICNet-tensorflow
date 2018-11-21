#!/usr/bin/env python

import sys
import cv2
from PIL import Image
from openvino.inference_engine import IENetwork, IEPlugin

model_xml='/opt/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/10_lrmodels/UNet/FP16/semanticsegmentation_frozen_person_32.xml'
model_bin='/opt/intel/computer_vision_sdk_2018.4.420/deployment_tools/model_optimizer/10_lrmodels/UNet/FP16/semanticsegmentation_frozen_person_32.bin'
net = IENetwork.from_ir(model=model_xml, weights=model_bin)
seg_image = Image.open("data/input/009649.png")
palette = seg_image.getpalette() # Get a color palette
index_void = 2 # Define index_void Back Ground
camera_width = 320
camera_height = 240

plugin = IEPlugin(device="HETERO:MYRIAD,CPU")
plugin.set_config({"TARGET_FALLBACK": "HETERO:MYRIAD,CPU"})
plugin.set_initial_affinity(net)
exec_net = plugin.load(network=net)

input_blob = next(iter(net.inputs))  #input_blob = 'input'
out_blob   = next(iter(net.outputs)) #out_blob   = 'conv6_cls/BiasAdd'
n, c, h, w = net.inputs[input_blob]  #n, c, h, w = 1, 3, 256, 512

del net

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    in_frame = cv2.resize(frame, (w, h))
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))

    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})

    if exec_net.requests[0].wait(-1) == 0:
        outputs = exec_net.requests[0].outputs[out_blob]

        # View
        output = outputs[0]
        res = np.argmax(output, axis=2)
        if index_void is not None:
            res = np.where(res == index_void, 0, res)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palette)
        image = image.convert("RGB")
        image = image.resize((camera_width, camera_height))

    cv2.imshow("Result", np.asarray(image))

    if if cv2.waitKey(1)&0xFF == ord('q'):
        break

cv2.destroyAllWindows()
del exec_net
del plugin
