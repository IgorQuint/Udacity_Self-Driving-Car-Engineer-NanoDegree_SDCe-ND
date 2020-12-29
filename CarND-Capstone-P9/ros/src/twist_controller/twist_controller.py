# Let's import all the libraries and modules I will be using for this
import rospy
# First the Yaw controller
from yaw_controller import YawController
# The PID controller
from pid import PID
# The Low Pass Filter
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self,vehicle_mass, fuel_capacity , brake_deadband, decel_limit, accel_limit, 
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel,max_steer_angle):
        # TODO: Implement
        # I will be using the provided controllers for now
        min_speed = 0.1
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)
        
        # Now, for the PID controller I need the Proportional, Integer and Derivative constants
        kp = 0.3
        ki = 0.1
        kd = 0.
        # Also, I need the min and max values, in this case for Throttle
        min_throttle = 0.
        max_throttle = 0.2
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        
        # Now the lowpass filter for high frequency noise
        tau = 0.5 #1/(2pi*tau) = cutoff frequency
        ts = 0.02 #sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()
        
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # First, I dbw_enabled is not True, the car will NOT be controlled by the controller.
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        # use the Low Pass Filter in the current velocity
        current_vel = self.vel_lpf.filt(current_vel)
        
        # Time to control the car Steering using the defined yaw controller
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        #let's see if it works
        #return 1., 0., steering
        
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        
        # If the "goal" velocity is 0 (maybe a red traffic light), I will send a brake transition
        # First, If I am near 0 velocity, apply the MAX brake force to make the car totally stops
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700 # N*m this is the required torque necessaty for Carla to totally brake.
        # If I am still going to fast, I will slowly decrese the car velocity    
        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit) # make the car not Hard Brake, too uncomfortable
            brake = abs(decel) * self.vehicle_mass*self.wheel_radius # Torque N*m
            
        return throttle, brake, steering
                 
        #return 1., 0., 0.  #Forcing values to see if the car moves in the simulator, just to test