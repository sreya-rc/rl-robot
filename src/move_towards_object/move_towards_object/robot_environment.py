import numpy as np

class RobotEnvironment:
    def __init__(self):
        self.state_dim = 3  # [forward_depth, current_rotation, person_detected]
        self.action_dim = 2  # [linear_velocity, angular_velocity]
        self.robot_start_position = np.array([0.00, -0.95, 0.14])
        self.current_position = self.robot_start_position.copy()
        self.current_rotation = 0.0
        self.wall_threshold = 0.3  # Distance threshold for wall detection
        self.safe_distance = 0.5   # Distance to consider safe for forward movement
        self.depth_reading = float('inf')  # Current depth sensor reading
        self.turning = False
        self.turn_increment = np.pi / 18  # 10-degree turn increment
        self.total_turn = 0.0  # Track total rotation during turning
        self.person_detected = False

    def reset(self):
        self.current_position = self.robot_start_position.copy()
        self.current_rotation = 0.0
        self.turning = False
        self.total_turn = 0.0
        self.depth_reading = float('inf')
        self.person_detected = False
        return self.calculate_state()

    def update_depth_reading(self, depth):
        self.depth_reading = depth

    def step(self, action):
        linear_vel, angular_vel = action

        if self.depth_reading < self.wall_threshold:
            self.turning = True
            self.total_turn = 0.0

        if self.turning:
            linear_vel = 0.0
            if abs(self.total_turn) < np.pi / 2:  # Less than 90 degrees
                angular_vel = self.turn_increment if self.total_turn >= 0 else -self.turn_increment
                self.total_turn += angular_vel
            else:
                self.turning = False
                self.total_turn = 0.0
        elif self.depth_reading > self.safe_distance:
            linear_vel = max(linear_vel, 0.2)  # Bias for forward movement
            angular_vel *= 0.5  # Reduce turning when safe

        self.current_position[0] += linear_vel * np.cos(self.current_rotation)
        self.current_position[1] += linear_vel * np.sin(self.current_rotation)
        self.current_rotation += angular_vel
        
        # Normalize rotation
        self.current_rotation = np.arctan2(np.sin(self.current_rotation), np.cos(self.current_rotation))

        new_state = self.calculate_state()
        reward = self.calculate_reward(linear_vel, angular_vel)
        done = self.person_detected

        return new_state, reward, done

    def calculate_state(self):
        return np.array([self.depth_reading, self.current_rotation, float(self.person_detected)])

    def calculate_reward(self, linear_vel, angular_vel):
        reward = 0.0
        if self.depth_reading > self.safe_distance:
            reward += linear_vel * 2.0  # Higher reward for forward motion when safe
            reward -= abs(angular_vel) * 0.5  # Small penalty for unnecessary turning
        elif self.depth_reading > self.wall_threshold:
            reward += linear_vel  # Smaller reward for careful forward motion
        if self.person_detected:
            reward += 10.0  # Large reward for detecting a person
        return reward

    def is_goal_reached(self):
        return self.person_detected