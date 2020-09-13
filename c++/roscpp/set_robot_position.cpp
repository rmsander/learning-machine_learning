#include <ros/ros.h>
#include "gazebo_msgs/SetModelState.h"

// Source: https://answers.ros.org/question/12116/calling-gazebo-service-set_model_state-in-c-code/ (User profile: https://answers.ros.org/users/189/sam/)
// NOTE: Works with ROS Kinetic and Gazebo 7.0.0 and Catkin Tools. Remember to add "gazebo_msgs" into CMakeLists.txt and package.xml as dependencies. (credits: https://answers.ros.org/users/5055/zheng/)//

using namespace std;

int main(int argc,char **argv)
{
    // ROS Client
    ros::init(argc,argv,"move_pr2_by_magic_node");

    //gazebo set_model_state
    ros::NodeHandle n;
    ros::ServiceClient client = n.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");

    // Position
    geometry_msgs::Point position;
    pr2_position.x = 1.0;
    pr2_position.y = 0.0;
    pr2_position.z = 0.0;
    
    // Orientation
    geometry_msgs::Quaternion orientation;
    pr2_orientation.x = 0.0;
    pr2_orientation.y = 0.0;
    pr2_orientation.z = 0.0;
    pr2_orientation.w = 1.0;

    // Pose (Pose + Orientation)
    geometry_msgs::Pose pr2_pose;
    pr2_pose.position = pr2_position;
    pr2_pose.orientation = pr2_orientation;

    // ModelState
    gazebo_msgs::ModelState modelstate;
    modelstate.model_name = (std::string) "model_state";
    modelstate.pose = pr2_pose;

    // Set Model State
    gazebo_msgs::SetModelState srv;
    srv.request.model_state = model_state;

    // Server Call

    if(client.call(srv))
    {
        ROS_INFO("Successfully moved robot!");
    }
    else
    {
        ROS_ERROR("Failed to move robot.! Error msg:%s",srv.response.status_message.c_str());
    }
    return 0;
}
