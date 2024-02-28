import rosbag2_py
import numpy as np

class Log():
    def __init__(self, filepath, topics=["/imu/data", "/teleop", "/vicon"]):
        self.filepath = filepath

        self.reader = rosbag2_py.SequentialReader()
        storage_options, converter_options = self.get_rosbag_options(self.filepath, "sqlite3")
        self.reader.open(storage_options, converter_options)
        self.reader.set_filter(rosbag2_py.StorageFilter(topics))

        topic_types = self.reader.get_all_topics_and_types()
        self.type_map = {
            topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
        }

        self.topic_data_map = {topic: np.array([]) for topic in topics}
        self.read()


    def read(self):
        print(self.type_map)


    def get_rosbag_options(self, path, storage_id, serialization_format='cdr'):
        storage_options = rosbag2_py.StorageOptions(
            uri=path, storage_id=storage_id)

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format)

        return storage_options, converter_options