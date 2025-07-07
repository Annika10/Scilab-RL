import numpy as np
import torch
import math
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_position_and_object_positions_of_observation(obs: torch.Tensor,
                                                     maximum_number_of_objects: int = 10,
                                                     observation_width: int = 10,
                                                     observation_height: int = 10,
                                                     agent_size: int = 1) -> torch.Tensor:
    """
    Get the position of the agent and up to maximum_number_of_objects objects in the observation.
    Args:
        obs: observation of size (batch_size, observation_width * observation_height)
        maximum_number_of_objects: the number of objects that are considered in the observation
        observation_width: width of the observation
        observation_height: height of the observation
        agent_size: size of the agent in the observation
        # FIXME: note, these are the first objects you get, when going down in the obs and going from left to right
        # FIXME: these are not necessary the nearest objects to the agent

    Returns:
        position of the agent and maximum_number_of_objects objects in the observation, where the first two elements
        are the x and y position of the agent
    """
    agent_and_object_positions = []
    if agent_size > 2:
        raise ValueError(f"Get the positions of an observation is only supported for agent size <= 2, "
                         f"but you defined an agent size of {agent_size}.")
    if not obs.shape[1] == (observation_width + 2) * observation_height:
        raise ValueError(
            f"The given observation width {observation_width} and height {observation_height} "
            f"do not match the observation shape: {obs.shape}."
            f"The second observation shape element {obs.shape[1]} should be "
            f"(observation_width + 2) * observation_height = {(observation_width + 2) * observation_height}.")
    for obs_element in obs:
        # agent in observation is marked with 1
        first_index_with_one = np.where(obs_element.cpu() == 1)[0][0]
        
        # object in observation is marked with 2 or 3
        if 2 in obs_element or 3 in obs_element:
            if 2 in obs_element:
                search_value = 2
            else:
                search_value = 3
            x_y_coordinates = []
            
            if agent_size == 1:
                indices_with_two_or_three = np.where(obs_element.cpu() == search_value)[0]
            # agent size is 2
            else:
                # Find indices where three consecutive ones occur
                indices_with_two_or_three = np.where(
                    ((obs_element.cpu()[:-2] == search_value) | (obs_element.cpu()[:-2] == 1))
                    & ((obs_element.cpu()[1:-1] == search_value) | (obs_element.cpu()[1:-1] == 1))
                    & ((obs_element.cpu()[2:] == search_value) | (obs_element.cpu()[2:] == 1))
                )[0]
                mask = (
                    # check if there is a line above
                        (np.isin(indices_with_two_or_three - (observation_width + 2), indices_with_two_or_three)
                         # and check if there is a line below
                         & np.isin(indices_with_two_or_three + (observation_width + 2), indices_with_two_or_three))
                        # otherwise check if the object is flying out of the grid (index in first line & not an object line two rows below)
                        | ((indices_with_two_or_three < observation_width + 2)
                           & (np.isin(indices_with_two_or_three + ((observation_width + 2) * 2),
                                      indices_with_two_or_three,
                                      invert=True)))
                        # otherwise check if the object is flying into the grid (index in last line) & not two lines above (then it is covered above)
                        | ((indices_with_two_or_three >= (observation_width + 2) * (observation_height - 1))
                           & (np.isin(indices_with_two_or_three - ((observation_width + 2) * 2),
                                      indices_with_two_or_three, invert=True)))
                )
                indices_with_two_or_three = indices_with_two_or_three[mask]
            
            for index in indices_with_two_or_three:
                
                # get x and y coordinate of object
                # +2 because of the walls
                x_coordinate = (index % (observation_width + 2)) + agent_size - 1
                # get to the middle of the object
                y_coordinate = math.floor(index / (observation_width + 2))
                
                if agent_size == 2:
                    # check if object is only in the first line (then y_coordinate is -1) or also in the second line
                    if y_coordinate == 0:
                        # check if second line has also an object at the x position
                        if not (torch.all(
                                (obs_element[
                                 (x_coordinate + observation_width + 1):(x_coordinate + observation_width + 4)]
                                 == search_value) | (obs_element[
                                                     (x_coordinate + observation_width + 1):(
                                                             x_coordinate + observation_width + 4)] == 1))):
                            y_coordinate = -1
                    elif y_coordinate == (observation_height - 1):
                        if not (torch.all(
                                (obs_element[((x_coordinate - 1) + (observation_width + 2) * (observation_height - 2))
                                :((x_coordinate + 2) + (observation_width + 2) * (observation_height - 2))]
                                 == search_value) |
                                (obs_element[((x_coordinate - 1) + (observation_width + 2) * (observation_height - 2))
                                :((x_coordinate + 2) + (observation_width + 2) * (observation_height - 2))] == 1))):
                            y_coordinate = observation_height
                
                # remove agent from indices with two or three --> agent is added later at the beginning of the list
                if not (x_coordinate == (first_index_with_one + agent_size - 1) and y_coordinate == (agent_size - 1)):
                    x_y_coordinates.append([x_coordinate, y_coordinate])
            
            x_y_coordinates_copy = copy.deepcopy(x_y_coordinates)
            for current_x_coordinate, current_y_coordinate in x_y_coordinates:
                # check if object coordinate is overlapping with the agent at the same y position
                if (
                        (
                                ((current_x_coordinate - 1) == (first_index_with_one + agent_size - 2)) or
                                ((current_x_coordinate - 1) == (first_index_with_one + agent_size - 1)) or
                                ((current_x_coordinate - 1) == (first_index_with_one + agent_size)) or
                                (current_x_coordinate == (first_index_with_one + agent_size - 2)) or
                                (current_x_coordinate == (first_index_with_one + agent_size - 1)) or
                                (current_x_coordinate == (first_index_with_one + agent_size)) or
                                ((current_x_coordinate + 1) == (first_index_with_one + agent_size - 2)) or
                                ((current_x_coordinate + 1) == (first_index_with_one + agent_size - 1)) or
                                ((current_x_coordinate + 1) == (first_index_with_one + agent_size))
                        )):
                    # object is left from agent
                    if current_x_coordinate < (first_index_with_one + agent_size - 1):
                        # check if object already started earlier
                        if obs_element.cpu()[
                            current_x_coordinate - 2 + (observation_width + 2) * min(max(current_y_coordinate, 0),
                                                                                     29)] == search_value:
                            x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])
                    elif current_x_coordinate > (first_index_with_one + agent_size - 1):
                        if obs_element.cpu()[
                            current_x_coordinate + 2 + (observation_width + 2) * min(max(current_y_coordinate, 0),
                                                                                     29)] == search_value:
                            x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])
                
                # check if object coordinate is overlapping with the agent at the same x position
                if (
                        (
                                ((current_y_coordinate - 1) == (agent_size - 2)) or
                                ((current_y_coordinate - 1) == (agent_size - 1)) or
                                ((current_y_coordinate - 1) == agent_size) or
                                (current_y_coordinate == (agent_size - 2)) or
                                (current_y_coordinate == (agent_size - 1)) or
                                (current_y_coordinate == agent_size) or
                                ((current_y_coordinate + 1) == (agent_size - 2)) or
                                ((current_y_coordinate + 1) == (agent_size - 1)) or
                                ((current_y_coordinate + 1) == agent_size)
                        ) and
                        (
                                current_x_coordinate == (first_index_with_one + agent_size - 1)
                        )
                ):
                    # object is below agent
                    if obs_element.cpu()[
                        ((current_y_coordinate + 2) * (observation_width + 2)) + current_x_coordinate] == search_value:
                        x_y_coordinates_copy.remove([current_x_coordinate, current_y_coordinate])
            
            x_y_coordinates = x_y_coordinates_copy
            # add zeros to the list if we have not enough objects
            if len(x_y_coordinates) < maximum_number_of_objects:
                x_y_coordinates = x_y_coordinates + [[0, 0]] * (maximum_number_of_objects - len(x_y_coordinates))
            # else: remove objects if we have too many objects
            elif len(x_y_coordinates) > maximum_number_of_objects:
                x_y_coordinates = x_y_coordinates[:maximum_number_of_objects]
            
            # add agent to object positions
            x_y_coordinates = [[first_index_with_one + agent_size - 1, agent_size - 1]] + x_y_coordinates
            agent_and_object_positions.append(x_y_coordinates)
        else:  # no object in observation
            # add agent to object positions + zeros for objects
            x_y_coordinates = [[first_index_with_one + agent_size - 1, agent_size - 1]] + [[0, 0] for i in
                                                                                           range(
                                                                                               maximum_number_of_objects)]
            agent_and_object_positions.append(x_y_coordinates)
    
    agent_and_object_positions_tensor = torch.flatten(
        torch.tensor(agent_and_object_positions, device=device, dtype=torch.float32),
        start_dim=1)
    
    return agent_and_object_positions_tensor


# outdated
def get_next_whole_observation(observations: torch.Tensor, actions: torch.Tensor, observation_width: int,
                               observation_height: int) -> torch.Tensor:
    """
    Calculate the next observation in the moonlander environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions
        observation_width: width of the observation
        observation_height: height of the observation

    Returns:
        next observation in the moonlander environment without input noise

    """
    raise NotImplementedError("This function is outdated.")
    if not observations.shape[1] == (observation_width + 2) * observation_height:
        raise ValueError(
            f"The given observation width {observation_width} and height {observation_height} "
            f"do not match the observation shape: {observations.shape}."
            f"The second observation shape element {observations.shape[1]} should be "
            f"(observation_width + 2) * observation_height = {(observation_width + 2) * observation_height}.")
    
    # object in observation is marked with 2 or 3
    if 2 in observations:
        search_value = 2
    else:
        search_value = 3
    
    # deep copy of next_observation
    observations_copy = observations.detach().clone()
    # remove agent and objects
    observations_copy[observations_copy == 1] = 0
    observations_copy[observations_copy == 2] = 0
    
    # get x coordinates of agent for each batch element
    x_index_of_agent = torch.nonzero(observations == 1, as_tuple=True)[1]
    # get x, y coordinates of objects for each batch element (index of batch element in element_in_batch)
    element_in_batch, indices_of_objects = torch.nonzero(observations == search_value, as_tuple=True)
    # get x and y coordinate of object
    x_coordinate_tensor = indices_of_objects % (observation_width + 2)
    y_coordinate_tensor = torch.floor(indices_of_objects / (observation_height + 2))
    
    new_y_coordinate_tensor = y_coordinate_tensor - 1
    new_indices_of_objects = ((new_y_coordinate_tensor * (observation_width + 2)) + x_coordinate_tensor).int()
    
    # remove negative y coordinates (object flew out of the grid)
    valid_mask = new_indices_of_objects >= 0
    valid_element_in_batch = element_in_batch[valid_mask]
    valid_new_indices_of_objects = new_indices_of_objects[valid_mask]
    
    # add objects to next observation
    observations_copy[valid_element_in_batch, valid_new_indices_of_objects] = 2
    
    # add agent to next observation
    new_x_index_of_agent = torch.clamp(x_index_of_agent + (actions - 1), min=1, max=observation_width)
    observations_copy[torch.arange(observations.shape[0]), new_x_index_of_agent] = 1
    
    return observations_copy


def get_observation_of_position_and_object_positions(agent_and_object_positions: torch.Tensor, observation_height: int,
                                                     observation_width: int, agent_size: int,
                                                     task: str) -> torch.Tensor:
    observations = []
    for agent_and_object_position in agent_and_object_positions:
        # tensor of (1,12) or (1, 1260)
        copy_of_agent_and_object_position = agent_and_object_position.clone().detach()
        # build empty obs
        matrix = np.zeros(shape=(observation_height, observation_width + 2), dtype=np.int16)
        
        if task == "dodge":
            object_value = 3
        elif task == "collect":
            object_value = 2
        else:
            raise ValueError("Task not supported.")
        
        # add objects
        counter = 2
        while counter < len(copy_of_agent_and_object_position):
            x_position_of_object = int(torch.round(copy_of_agent_and_object_position[counter]))
            y_position_of_object = int(torch.round(copy_of_agent_and_object_position[counter + 1]))
            if (agent_size <= x_position_of_object <= observation_width + 1 - agent_size) and (
                    -agent_size + 1 <= y_position_of_object <= observation_height):
                matrix[
                # objects can also fly into the grid
                max(
                    0,
                    min(
                        observation_height - 1,
                        int(torch.round(copy_of_agent_and_object_position[counter + 1]) - agent_size + 1)
                    )
                ):  # y start of object
                # objects can also fly out of the grid
                max(
                    0,
                    min(
                        observation_height - 1,
                        int(torch.round(copy_of_agent_and_object_position[counter + 1]) + agent_size - 1)
                    )
                ) + 1,  # y end of object
                x_position_of_object - agent_size + 1:  # x start of object
                x_position_of_object + agent_size  # x end of object
                ] = object_value
            counter += 2
        
        # add agent
        x_position_of_agent = int(torch.round(copy_of_agent_and_object_position[0]))
        # make sure that the agent is within the matrix
        if x_position_of_agent < agent_size:
            x_position_of_agent = agent_size
        elif x_position_of_agent > observation_width + 1 - agent_size:
            x_position_of_agent = observation_width + 1 - agent_size
        
        # first element is the y position of the agent, second element is the x position of the agent
        matrix[
        max(0, min(observation_height - (2 * agent_size - 1),
                   int(torch.round(copy_of_agent_and_object_position[1]) - agent_size + 1))):  # y start of agent
        max(2 * agent_size - 2,
            min(observation_height - 1,
                int(torch.round(copy_of_agent_and_object_position[1]) + agent_size - 1))) + 1,
        # y end of agent
        x_position_of_agent - agent_size + 1:  # x start of agent
        x_position_of_agent + agent_size] = 1  # x end of agent
        
        # add wall
        matrix[:, 0] = -1
        matrix[:, -1] = -1
        
        observations.append(matrix)
    
    # form observations list to numpy array for faster calculations
    observations = np.array(observations)
    
    observations_tensor = torch.flatten(torch.tensor(observations, device=device, dtype=torch.float32), start_dim=1)
    
    return observations_tensor


def get_next_position_observation_moonlander(observations: torch.Tensor, actions: torch.Tensor, observation_width: int,
                                             agent_size: int) -> torch.Tensor:
    """
    Calculate the next observation in the moonlander environment manually to exclude random observations through input noise.
    Args:
        observations: observations
        actions: actions
        observation_width: width of the observation
        agent_size: size of the agent in the observation

    Returns:
        next observation in the moonlander environment without input noise
    """
    next_observation_without_input_noise = observations.clone().detach().to(device)
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions).clone().detach().squeeze(dim=0).to(device)
    
    # loop through every observation in batch
    for index, obs in enumerate(observations):
        # first two elements are agent x and y position
        counter = 2
        # index[0] is agent x position
        # apply action to agent x position (actions: 0, 1, 2)
        # action_movement is -1 to go left, 0 to stay and 1 to go right for agent size 1
        # for agent size 2 it is -2, 0, 2
        # for agent size 3 it is -3, 0, 3 ...
        next_observation_without_input_noise[index][0] += (agent_size * actions[index] - agent_size)
        # clip to range of agent_size to observation_width
        next_observation_without_input_noise[index][0] = torch.clamp(next_observation_without_input_noise[index][0],
                                                                     agent_size, observation_width - agent_size + 1)
        
        # apply step to every object y position
        next_observation_without_input_noise[index][3::2] -= 1
        
        # check if there is an object that already is now on position -2 (for agent size 2) or -3 (for agent size 3)
        # after doing a step -> remove it
        while counter <= (observations.shape[1] - 2):
            if not next_observation_without_input_noise[index][counter] == 0 and \
                    next_observation_without_input_noise[index][counter + 1] == -agent_size:
                next_observation_without_input_noise[index][counter] = 0
                next_observation_without_input_noise[index][counter + 1] = 0
            # clamp y-position of empty object positions back to 0
            elif next_observation_without_input_noise[index][counter] == 0 and \
                    next_observation_without_input_noise[index][counter + 1] == -1:
                next_observation_without_input_noise[index][counter + 1] = 0
            counter += 2
    
    return next_observation_without_input_noise


def get_collected_objects(observation_positions: torch.tensor, agent_size: int, observation_width: int) -> list[
    dict[str, int]]:
    if agent_size > 2:
        raise NotImplementedError("Agent size > 2 is not supported to get the collected objects in the positions.")
    x_position_of_agent = int(
        min(max(agent_size, observation_positions[0][0]), observation_width - agent_size + 1))
    y_position_of_agent = int(observation_positions[0][1])
    collected_objects = []
    
    for index in range(2, len(observation_positions[0]), 2):
        if not (observation_positions[0][index] == 0 and observation_positions[0][index + 1] == 0):
            
            if agent_size == 1:
                if (observation_positions[0][index] == x_position_of_agent) and (
                        observation_positions[0][index + 1] == y_position_of_agent):
                    collected_objects.append(
                        {'x': int(observation_positions[0][index]),
                         'y': int(observation_positions[0][index + 1]),
                         'size': agent_size})
            # agent size of 2
            else:
                if (
                        (
                                ((observation_positions[0][index] - 1) == (x_position_of_agent - 1))
                                or ((observation_positions[0][index] - 1) == x_position_of_agent)
                                or ((observation_positions[0][index] - 1) == (x_position_of_agent + 1))
                                or (observation_positions[0][index] == (x_position_of_agent - 1))
                                or (observation_positions[0][index] == x_position_of_agent)
                                or (observation_positions[0][index] == (x_position_of_agent + 1))
                                or ((observation_positions[0][index] + 1) == (x_position_of_agent - 1))
                                or ((observation_positions[0][index] + 1) == x_position_of_agent)
                                or ((observation_positions[0][index] + 1) == (x_position_of_agent + 1))
                        )
                        and
                        (
                                ((observation_positions[0][index + 1] - 1) == (y_position_of_agent - 1))
                                or ((observation_positions[0][index + 1] - 1) == y_position_of_agent)
                                or ((observation_positions[0][index + 1] - 1) == (y_position_of_agent + 1))
                                or (observation_positions[0][index + 1] == (y_position_of_agent - 1))
                                or (observation_positions[0][index + 1] == y_position_of_agent)
                                or (observation_positions[0][index + 1] == (y_position_of_agent + 1))
                                or ((observation_positions[0][index + 1] + 1) == (y_position_of_agent - 1))
                                or ((observation_positions[0][index + 1] + 1) == y_position_of_agent)
                                or ((observation_positions[0][index + 1] + 1) == (y_position_of_agent + 1))
                        )
                ):
                    collected_objects.append(
                        {'x': int(observation_positions[0][index]),
                         'y': int(observation_positions[0][index + 1]),
                         'size': agent_size})
    
    return collected_objects
