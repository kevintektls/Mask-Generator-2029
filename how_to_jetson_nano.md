Plug micro sd card (with OS) into the jetson nano's micro sd port (at the back)
Plug in TP-LINK (wifi) + keyboard + mouse + screen
Connect to network on jetson nano
From laptop/computer connect to the jetson nano through ssh (ssh robotcar@<jetson-ip>) (jetson-ip can be found by running 'hostname -i')
Once connected through ssh, you can remove screen, mouse and keyboard (we need as much power as possible)

[Camera]
Plug in camera (OAK-D Lite) and run the 'camera_website.py' script found in camera_scripts/ (python3 camera_scripts/camera_website.py <target fps>)
Connect from your laptop/computer to <jetson-ip>:8080

[VNC]
Run in jetson nano terminal: 'x11vnc -display :0 -forever -nopw -listen 0.0.0.0 -rfbport 5900'
Open any VNC Viewer on your laptop/computer
Connect to <jetson-ip>:5900
