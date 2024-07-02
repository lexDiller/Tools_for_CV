import cv2


def trim_and_concat_videos(input_file, output_file, intervals):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(input_file)
    out = None
    for start_time, end_time in intervals:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_frame_time >= start_time:
                if out is None:
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                    out = cv2.VideoWriter(output_file, fourcc, out_frame_rate, (frame_width, frame_height))
                out.write(frame)

            if current_frame_time >= end_time:
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def calculate_seconds(minutes, seconds):
    return (minutes * 60) + seconds


def input_intervals():
    intervals = []
    while True:
        try:
            minutes_input = input("Minutes ('Enter' for exit): ")
            if minutes_input == "":
                break
            minutes = int(minutes_input)

            seconds = int(input("Seconds: "))

            total_seconds = calculate_seconds(minutes, seconds)
            intervals.append(total_seconds)
        except ValueError:
            print("Incorrect values")

    if len(intervals) % 2 != 0:
        print("The odd number of intervals, need one more")
        return input_intervals()

    tuple_list = [(intervals[i], intervals[i + 1]) for i in range(0, len(intervals), 2)]
    return tuple_list


def main(input_video, output_video):
    intervals = input_intervals()
    trim_and_concat_videos(input_video, output_video, intervals)


if __name__ == "__main__":
    main(input_video='/path', output_video='/path')
