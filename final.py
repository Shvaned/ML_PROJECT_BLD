import matplotlib.pyplot as plt


# Function to read and parse the classes.txt file
def read_probabilities_from_file(file_path):
    probabilities = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Extract the probability from the line
                prob = float(line.strip())  # Convert the probability string to float
                probabilities.append(prob)
            except ValueError:
                continue  # Ignore any lines that do not contain a valid probability
    return probabilities


# Function to classify probabilities into high (1) or low (0)
def classify_class(probability):
    return 1 if probability >= 37.5 else 0


# Define the conclusions for all possible combinations
conclusions = {
    '0000': 'The target is likely in a low-stimulation environment.',
    '0001': 'The target is showing composure, likely in a relaxed or controlled environment.',
    '0010': 'The target appears to be focused on a task.',
    '0011': 'The target appears composed and focused.',
    '0100': 'The target seems to be occupied.',
    '0101': 'The target is both composed in a group.',
    '0110': 'The target seems to be engaged in a focused task while in a group.',
    '0111': 'The target is composed and focused, but in a group.',
    '1000': 'The target is being observant and aware.',
    '1001': 'The target is being observant in a controlled setting.',
    '1010': 'The target is observant and in a high-pressure situation amongst others.',
    '1011': 'The target is being hasty yet composed and focused.',
    '1100': 'The target is focused and observant amongst others.',
    '1101': 'The target is observant and composed amongst others.',
    '1110': 'The target is observant, focused, and hasty amongst others.',
}

# Define the additional conclusions based on binary classification
additional_conclusions = {
    '0000': {
        'Work Environment': 'The target seems to be working in a quiet, individual workspace, possibly doing focused work with little distractions.',
        'Job Interviews': 'The target is likely waiting in a quiet, controlled environment before an interview.',
        'Work Meetings': 'The target is possibly in a calm environment, not yet actively participating in the meeting.',
        'Social Gathering': 'The target may be standing alone or in a passive role at a social event.',
        'Public Environment': 'The target is likely in a calm public space, perhaps observing without much interaction.',
    },
    '0001': {
        'Work Environment': 'The target appears composed, working in a stable and controlled office environment.',
        'Job Interviews': 'The target is likely calm and composed in a waiting area or preparatory stage of an interview.',
        'Work Meetings': 'The target is sitting calmly, absorbing information in a professional meeting setting.',
        'Social Gathering': 'The target is calm and composed, potentially in a quiet corner of a social gathering.',
        'Public Environment': 'The target remains composed in a busy, public space, potentially scanning the surroundings quietly.',
    },
    '0010': {
        'Work Environment': 'The target is deeply focused on a task, possibly alone and immersed in work-related activities.',
        'Job Interviews': 'The target may be focusing intently on answering questions during a job interview.',
        'Work Meetings': 'The target is attentively participating in a meeting, likely focusing on discussions or presentations.',
        'Social Gathering': 'The target seems focused on a conversation or small task in a social event.',
        'Public Environment': 'The target is observing or focusing on a specific element in a public environment.',
    },
    '0011': {
        'Work Environment': 'The target is focused and calm in a controlled, productive workspace.',
        'Job Interviews': 'The target is maintaining focus and composure while answering questions in an interview.',
        'Work Meetings': 'The target is composed and attentive, contributing effectively in a formal work meeting.',
        'Social Gathering': 'The target seems calm and focused, possibly engaged in an important conversation at a gathering.',
        'Public Environment': 'The target is composed and focused, potentially scanning for specific information or events in public.',
    },
    '0100': {
        'Work Environment': 'The target is occupied with work, possibly engaged in a task or meeting that requires their attention.',
        'Job Interviews': 'The target may be waiting in a semi-occupied environment, preparing for an upcoming interview.',
        'Work Meetings': 'The target is likely engaged in a meeting, possibly taking notes or reviewing materials.',
        'Social Gathering': 'The target seems involved in a group conversation or social interaction, contributing occasionally.',
        'Public Environment': 'The target is likely engaged with their surroundings or task in a public space, staying aware of stimuli.',
    },
    '0101': {
        'Work Environment': 'The target is engaged in a group work setting, maintaining composure while collaborating on a project.',
        'Job Interviews': 'The target is calm in a group interview setting, contributing their responses with composure.',
        'Work Meetings': 'The target appears composed while listening or contributing in a group meeting environment.',
        'Social Gathering': 'The target is participating in a social gathering while remaining calm, possibly mingling with others.',
        'Public Environment': 'The target seems calm and observant in a public environment, perhaps scanning a group or crowd.',
    },
    '0110': {
        'Work Environment': 'The target is likely focused on a task or discussion in a group work environment, contributing actively.',
        'Job Interviews': 'The target may be answering questions while managing attention in a group interview or panel discussion.',
        'Work Meetings': 'The target is actively engaged in a group meeting, providing feedback or focusing on the discussion.',
        'Social Gathering': 'The target seems to be focused on an important conversation while balancing group dynamics.',
        'Public Environment': 'The target may be actively engaged with a group while paying attention to their environment in a public space.',
    },
    '0111': {
        'Work Environment': 'The target is in a group setting, composed and contributing productively while maintaining focus on tasks.',
        'Job Interviews': 'The target remains composed and focused in a group interview, answering questions with poise.',
        'Work Meetings': 'The target shows both composure and focus in a group meeting, likely leading or participating effectively.',
        'Social Gathering': 'The target is maintaining focus and composure in a lively social gathering, balancing attention and interactions.',
        'Public Environment': 'The target is composed and focused in a public environment, possibly engaged in a group conversation or observation.',
    },
    '1000': {
        'Work Environment': 'The target is alert and aware of their surroundings, possibly scanning for work-related cues or tasks.',
        'Job Interviews': 'The target seems highly observant, likely watching for non-verbal cues in a job interview.',
        'Work Meetings': 'The target is scanning the room for relevant information, observing how the meeting unfolds.',
        'Social Gathering': 'The target appears alert, noticing body language and social cues in a social setting.',
        'Public Environment': 'The target is keenly aware of their environment, possibly watching out for certain events or stimuli in a public space.',
    },
    '1001': {
        'Work Environment': 'The target is observing the details of their environment, taking note of ongoing activities or cues.',
        'Job Interviews': 'The target is observant in a formal interview setting, potentially reading the room or interviewers’ reactions.',
        'Work Meetings': 'The target is closely watching the dynamics of a meeting, possibly picking up on details that others overlook.',
        'Social Gathering': 'The target is aware of social cues and interactions, quietly observing the flow of conversations.',
        'Public Environment': 'The target is observant in a controlled public environment, watching people or specific events closely.',
    },
    '1010': {
        'Work Environment': 'The target is in a high-pressure environment, staying observant while performing tasks or interacting with a team.',
        'Job Interviews': 'The target seems observant and focused during a high-pressure interview, staying alert to the interviewers’ expectations.',
        'Work Meetings': 'The target is absorbing the details of a tense meeting, observing everyone’s reactions carefully.',
        'Social Gathering': 'The target seems observant in a stressful social event, maintaining awareness of interactions or group dynamics.',
        'Public Environment': 'The target is observing their surroundings carefully, possibly responding to high-pressure stimuli in public.',
    },
    '1011': {
        'Work Environment': 'The target is working with urgency, maintaining focus and composure while managing time-sensitive tasks.',
        'Job Interviews': 'The target may be answering questions hastily yet remaining composed under interview pressure.',
        'Work Meetings': 'The target is engaging in a fast-paced meeting, keeping their focus while responding quickly to questions or tasks.',
        'Social Gathering': 'The target may be hasty in moving through the social gathering, maintaining composure despite the rush.',
        'Public Environment': 'The target is rushing through a busy public space, staying aware and composed amidst the crowd.',
    },
    '1100': {
        'Work Environment': 'The target is focused and observant amongst others, possibly monitoring colleagues or team activities during collaborative work.',
        'Job Interviews': 'The target seems observant and engaged in a group interview, paying attention to other candidates or the interviewers’ reactions.',
        'Work Meetings': 'The target is focused and observant, likely listening to others while contributing in a group meeting setting.',
        'Social Gathering': 'The target is engaged and aware, possibly observing others while participating in a conversation or group discussion at a social gathering.',
        'Public Environment': 'The target is focused and observant in a public space, possibly scanning the surroundings while maintaining awareness of their immediate environment.',
    },
}


# Main logic
def main():
    # Read probabilities from the file
    file_path = 'classes.txt'  # Path to your classes.txt file
    predicted_probabilities = read_probabilities_from_file(file_path)

    # Check that the number of probabilities is 4 (for the 4 classes)
    if len(predicted_probabilities) != 4:
        print("Error: Expected 4 probabilities, but found", len(predicted_probabilities))
        return

    # Classify each class based on the probabilities
    classifications = [classify_class(prob) for prob in predicted_probabilities]

    # Convert to binary string (e.g., "0010")
    binary_classification = ''.join(str(classification) for classification in classifications)

    # Output the conclusion for the given classification
    conclusion = conclusions.get(binary_classification, 'No conclusion available for this classification')

    # Output the additional context for the given classification
    additional = additional_conclusions.get(binary_classification, None)
    if additional:
        work_env = additional['Work Environment']
        job_interviews = additional['Job Interviews']
        work_meetings = additional['Work Meetings']
        social_gathering = additional['Social Gathering']
        public_env = additional['Public Environment']
    else:
        work_env = job_interviews = work_meetings = social_gathering = public_env = 'No additional conclusion available.'

    # Print conclusions
    print(f"General Conclusion: {conclusion}")
    print(f"Work Environment: {work_env}")
    print(f"Job Interviews: {job_interviews}")
    print(f"Work Meetings: {work_meetings}")
    print(f"Social Gathering: {social_gathering}")
    print(f"Public Environment: {public_env}")

    # Plotting the conclusions as an image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.1, 0.9, f"Conclusion: {conclusion}", fontsize=14, wrap=True)
    ax.text(0.1, 0.8, f"Work Environment: {work_env}", fontsize=12, wrap=True)
    ax.text(0.1, 0.7, f"Job Interviews: {job_interviews}", fontsize=12, wrap=True)
    ax.text(0.1, 0.6, f"Work Meetings: {work_meetings}", fontsize=12, wrap=True)
    ax.text(0.1, 0.5, f"Social Gathering: {social_gathering}", fontsize=12, wrap=True)
    ax.text(0.1, 0.4, f"Public Environment: {public_env}", fontsize=12, wrap=True)

    ax.axis('off')
    plt.show()


# Run the main function
if __name__ == '__main__':
    main()
