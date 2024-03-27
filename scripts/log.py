import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


class Log:
    def __init__(self, filepath, topics=None):
        self.filepath = filepath

        self.reader = rosbag2_py.SequentialReader()
        storage_options, converter_options = self.get_rosbag_options(
            self.filepath, "sqlite3"
        )
        self.reader.open(storage_options, converter_options)

        topic_types = self.reader.get_all_topics_and_types()
        self.type_map = {
            topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
        }

        if not topics:
            topics = [topic_types[i].name for i in range(len(topic_types))]

        self.reader.set_filter(rosbag2_py.StorageFilter(topics))
        self.topic_data_map = {topic: [] for topic in topics}
        self.read()

    def read(self):
        while self.reader.has_next():
            (topic, data, _) = self.reader.read_next()

            msg_type = get_message(self.type_map[topic])
            msg = deserialize_message(data, msg_type)

            self.topic_data_map[topic].append(msg)

    def get_rosbag_options(self, path, storage_id, serialization_format="cdr"):
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format,
        )

        return storage_options, converter_options
