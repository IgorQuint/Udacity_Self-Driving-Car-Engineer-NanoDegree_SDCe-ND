#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
# for the traffic_waypoint
from std_msgs.msg import Int32

# For locating the closest waypoint in space
from scipy.spatial import KDTree

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 80 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .4

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add other member variables you need below
        self.base_lane = None
        self.pose = None
        self.stopline_wp_idx = -1
        #self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # Traffic waypoint has a int 32 message type, so I need to import that message type above
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


        self.loop()
    
    # We use a loop fuction to have control over the publishing frequency
    def loop(self):
        rate = rospy.Rate(20) # I can try different rates, down to 20 Hz
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                # First, we have to be sure that we received the Car Position and Waypoints.
                
                # Get closest waypoint. This was for the Partial Waypoint Updater before using twist control.
                #closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints()
            rate.sleep()
            
     # Let's define the two functions I need in the loop function
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        if self.waypoint_tree:
            closest_idx = self.waypoint_tree.query([x, y], 1)[1]
            # Check to see if the point is ahead or behind
            closest_coord = self.waypoints_2d[closest_idx]
            prev_coord = self.waypoints_2d[closest_idx - 1]
      
            # Equation for hyperplane through closest_coords
            cl_vect = np.array(closest_coord)
            prev_vect = np.array(prev_coord)
            pos_vect = np.array([x, y])
        
            # Based on the explanation in the "Waypoint Updater Partial Walkthrough" min 10
            val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
            if val > 0:
                closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
            return closest_idx
    
    # The first method is responsible of Publishing the Waypoints generated in the generate_lane method.
    def publish_waypoints(self):
        #lane = Lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
    
    # This second method takes the closest waypoint and the number of waypoints ahead and generate a list of the waypoints the car has to follow
    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        
        # In case the car does not find a RED traffic light ahead, or the traffic light is far, we just add the base_waypoints to the car lane waypoints
        # stopline is -1 when the Traffic light is NOT red
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
            
        # If there IS a  RED traffic light ahead and near, we use the Method decelerate_waypoints to change the car behaviour
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane
    
    # This third method is used when there is a RED traffic light infront of our car.
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            # 
            stop_idx = max(self.stopline_wp_idx - closest_idx - 3, 0)
            dist = self.distance(waypoints, i, stop_idx)
            # A mathematical function to adjust the decreasing speed to the distance between the car and the stopping point. This is for the car to stop smoothly
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel < 1. :
                vel = 0.
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        
        return temp
    
    # Callback functions for the ROS topics
    def pose_cb(self, msg):
        # TODO: Implement
        # Just assign the incoming position data to the internal variable pose
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # Let's begin with just passing the coming waypoints
        self.base_lane = waypoints
        if not self.waypoints_2d:
            # get X and Y points from the waypoints and add them to a tree structure from KDTree
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        
        #self.base_waypoints_sub.unregister()
          
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
