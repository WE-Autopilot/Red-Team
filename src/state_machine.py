import keyboard
import time
from transitions import Machine
 
class F1TENTH_StopSignFSM:
    states = ["DRIVING", "SLOWING_DOWN", "STOPPED"]

    def __init__(self):
        self.machine = Machine(model=self, states=F1TENTH_StopSignFSM.states, initial="DRIVING")

        # State Transitions
        self.machine.add_transition("detect_stop_sign", "DRIVING", "SLOWING_DOWN", before="slow_down")
        self.machine.add_transition("stop", "SLOWING_DOWN", "STOPPED", before="apply_brakes")
        self.machine.add_transition("proceed", "STOPPED", "DRIVING", before="accelerate")

    def slow_down(self):
        
        print("Slowing down for stop sign...")

    def apply_brakes(self):
        print("Car has stopped.")

    def accelerate(self):
        print("Accelerating.")

# Initialize FSM
car_fsm = F1TENTH_StopSignFSM()

# Function to simulate a stop sign when "C" is pressed
def check_keyboard():
    if keyboard.is_pressed("c"):
        print("\n[DEBUG] Simulated stop sign detected!\n")
        car_fsm.detect_stop_sign()

# Main loop to listen for keypress and handle FSM transitions
while True:
    check_keyboard()

    # Handle transitions automatically
    if car_fsm.state == "SLOWING_DOWN":
        time.sleep(1)  # Simulate slowing down time
        car_fsm.apply_brakes()

    if car_fsm.state == "STOPPED":
        time.sleep(3)  # Stop for 3 seconds
        car_fsm.accelerate()

    time.sleep(0.1)  # Prevent CPU overuse
