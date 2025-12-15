import random
from typing import List, Tuple

def generate_q_a_pair(mode="ADD", sm_mode=0) -> List[str]:
    """Generate diverse Q/A pairs for LLM training with properly structured templates."""
    range_use_min = -9
    range_use_max = 9
    match sm_mode:
        case 0:
            pass
        case 1:
            range_use_min = -99
            range_use_max = 99
        case 2:
            range_use_min = -999
            range_use_max = 999
        case 3:
            range_use_min = -9999
            range_use_max = 9999
        case 4:
            range_use_min = -99999
            range_use_max = 99999
    
    # Generate random terms
    termA = random.randint(range_use_min, range_use_max)
    termB = random.randint(range_use_min, range_use_max)
    
    # For division, ensure non-zero denominator and avoid decimals for cleaner data
    if mode == "DIVIDE":
        while termB == 0:
            termB = random.randint(range_use_min, range_use_max)
        # Make division cleaner by adjusting numbers
        termA = termA * termB
    
    # Categorized question structures
    question_structures = [
        # (structure_template, requires_starter, description)
        ("{starter} {template}", True, "instruction_with_starter"),
        ("{template}?", False, "direct_question"),
        ("I need to {template}", False, "statement_format"),
        ("Can you help me {template}?", False, "conversational_help"),
        ("{template}", False, "bare_instruction"),
        ("What is {template}?", False, "what_is_format"),
        ("How do I calculate {template}?", False, "how_calculate_format"),
        ("Show me {template}", False, "show_me_format"),
    ]
    
    # Starters that require infinitive templates
    starters_requiring_infinitive = [
        "How do I", "how do i", "How can I", "how can i", 
        "What's the way to", "what's the way to", "Show me how to",
        "Can you explain how to", "Explain how to", "I need to know how to",
        "Tell me how to", "Could you show me how to", "Demonstrate how to",
        "Walk me through how to", "Help me understand how to"
    ]
    
    # Starters that work with any template type
    starters_any = [
        "Calculate", "calculate", "Compute", "compute", "Find", "find",
        "Determine", "determine", "Work out", "work out", "Solve", "solve",
        "What is", "what is", "What's", "what's"
    ]
    
    # Categorized templates by what they work with
    templates = {
        "ADD": {
            "infinitive": [  # Work with "how to" starters and "I need to"
                "add {a} and {b}", "add {b} to {a}", "add {a} & {b}",
                "perform addition on {a} and {b}", "do addition with {a} and {b}",
                "calculate the sum of {a} and {b}", "find the total of {a} and {b}",
                "compute {a} plus {b}", "work out {a} + {b}", "solve {a} + {b}",
                "get the sum for {a} and {b}", "combine {a} and {b} through addition",
                "put together {a} and {b}", "show the addition of {a} and {b}",
                "demonstrate adding {a} and {b}", "sum up {a} and {b}",
                "figure out {a} plus {b}"
            ],
            "direct_action": [  # Work with bare starters and direct formats
                "add {a} and {b}", "add {b} to {a}", "add {a} & {b}",
                "the sum of {a} and {b}", "{a} plus {b}", "{a} + {b}",
                "addition of {a} and {b}", "{a} added to {b}", "{b} added to {a}",
                "adding {a} and {b}"
            ],
            "question_form": [  # Work with "what is" and question formats
                "{a} plus {b}", "{a} + {b}", "the sum of {a} and {b}",
                "the total of {a} and {b}", "{a} added to {b}", 
                "if I add {b} to {a}", "what I get when I add {a} and {b}",
                "adding {b} to {a}", "the result of {a} + {b}"
            ],
            "conceptual": [  # Work with various formats
                "the sum of {a} and {b}", "the total when combining {a} and {b}",
                "what you get when adding {a} and {b}", "the result of adding {a} and {b}"
            ]
        },
        "SUB": {
            "infinitive": [
                "subtract {b} from {a}", "perform subtraction on {a} and {b}",
                "calculate {a} minus {b}", "find the difference between {a} and {b}",
                "compute {a} - {b}", "work out {a} - {b}", "solve {a} - {b}",
                "get the result of {a} minus {b}", "take away {b} from {a}",
                "deduct {b} from {a}", "remove {b} from {a}",
                "find how much larger {a} is than {b}", "show subtraction between {a} and {b}",
                "figure out {a} minus {b}"
            ],
            "direct_action": [
                "subtract {b} from {a}", "{a} minus {b}", "{a} - {b}",
                "the difference between {a} and {b}", "subtraction of {b} from {a}",
                "taking {b} from {a}"
            ],
            "question_form": [
                "{a} minus {b}", "{a} - {b}", "the difference between {a} and {b}",
                "if I subtract {b} from {a}", "what I get when I take {b} from {a}",
                "{a} take away {b}", "the result of {a} - {b}",
                "how much larger {a} is than {b}"
            ],
            "conceptual": [
                "the difference between {a} and {b}", "the result of subtracting {b} from {a}",
                "what remains when you take {b} from {a}"
            ]
        },
        "MULT": {
            "infinitive": [
                "multiply {a} by {b}", "multiply {b} by {a}", 
                "perform multiplication on {a} and {b}", "calculate the product of {a} and {b}",
                "compute {a} times {b}", "work out {a} × {b}", "solve {a} * {b}",
                "find {a} times {b}", "get the product for {a} and {b}",
                "show multiplication of {a} and {b}", "demonstrate multiplying {a} and {b}",
                "compute the product between {a} and {b}", "figure out {a} times {b}"
            ],
            "direct_action": [
                "multiply {a} by {b}", "{a} times {b}", "{a} × {b}", "{a} * {b}",
                "the product of {a} and {b}", "multiplication of {a} and {b}",
                "{a} multiplied by {b}"
            ],
            "question_form": [
                "{a} times {b}", "{a} × {b}", "{a} multiplied by {b}",
                "the product of {a} and {b}", "if I multiply {a} by {b}",
                "what I get when multiplying {a} and {b}", "the result of {a} × {b}",
                "what {a} times {b} is"
            ],
            "conceptual": [
                "the product of {a} and {b}", "the result of multiplying {a} by {b}",
                "what you get when you multiply {a} and {b}"
            ]
        },
        "DIVIDE": {
            "infinitive": [
                "divide {a} by {b}", "divide {b} into {a}", 
                "perform division on {a} and {b}", "calculate {a} divided by {b}",
                "compute {a} ÷ {b}", "work out {a} / {b}", "find the quotient of {a} and {b}",
                "get the result of {a} ÷ {b}", "show division of {a} by {b}",
                "demonstrate dividing {a} by {b}", "calculate the quotient for {a} divided by {b}",
                "figure out {a} divided by {b}", "work out the quotient of {a} and {b}"
            ],
            "direct_action": [
                "divide {a} by {b}", "{a} divided by {b}", "{a} ÷ {b}", "{a} / {b}",
                "the quotient of {a} and {b}", "division of {a} by {b}",
                "{a} over {b}"
            ],
            "question_form": [
                "{a} divided by {b}", "{a} ÷ {b}", "{a} / {b}", "{a} over {b}",
                "the quotient of {a} and {b}", "if I divide {a} by {b}",
                "what I get when dividing {a} by {b}", "the result of {a} ÷ {b}",
                "how many times {b} goes into {a}", "what {a} divided by {b} is"
            ],
            "conceptual": [
                "the quotient of {a} and {b}", "the result of dividing {a} by {b}",
                "what you get when you divide {a} by {b}"
            ]
        }
    }
    
    # Answer formats
    answer_formats = [
        "{a} {op} {b} = {result}",
        "The answer is {result}",
        "{a} {op} {b} equals {result}",
        "Result: {result}",
        "Calculation: {a} {op} {b} = {result}",
        "{result}",
        "You get {result}",
        "The solution is {result}",
        "After calculating, the result is {result}",
        "{a} {op} {b} gives you {result}",
        "That would be {result}",
        "It's {result}",
        "The result is {result}",
        "I get {result}",
        "That equals {result}",
        "{result} is the answer",
    ]
    
    # Operation symbols
    op_symbols = {
        "ADD": ["+", "plus"],
        "SUB": ["-", "minus"], 
        "MULT": ["×", "*", "times", "multiplied by"],
        "DIVIDE": ["÷", "/", "divided by"]
    }
    
    # Select question structure
    structure_template, requires_starter, structure_type = random.choice(question_structures)
    
    # Choose appropriate starter and template category
    if requires_starter:
        if structure_type == "instruction_with_starter":
            # Use infinitive starters with infinitive templates
            starter = random.choice(starters_requiring_infinitive)
            template_category = "infinitive"
        else:
            starter = random.choice(starters_any + starters_requiring_infinitive)
            template_category = random.choice(["infinitive", "direct_action", "question_form"])
    else:
        starter = ""
        if structure_type == "direct_question":
            template_category = "question_form"
        elif structure_type == "statement_format":
            template_category = "infinitive"
        elif structure_type in ["what_is_format", "how_calculate_format"]:
            template_category = "conceptual"
        else:
            template_category = random.choice(["direct_action", "question_form", "conceptual"])
    
    # Get appropriate template
    template = random.choice(templates[mode][template_category])
    
    # Smart term ordering - don't swap for templates that specify order
    should_swap = (
        random.random() > 0.5 and 
        template_category != "infinitive" and
        not any(phrase in template for phrase in ["from", "into", "away from", "subtract"])
    )
    
    if should_swap:
        termA, termB = termB, termA
    
    # Calculate result
    operations = {
        "ADD": lambda x, y: x + y,
        "SUB": lambda x, y: x - y, 
        "MULT": lambda x, y: x * y,
        "DIVIDE": lambda x, y: x // y if y != 0 else "undefined"
    }
    
    result = operations[mode](termA, termB)
    
    # Format question
    if requires_starter:
        question = structure_template.format(starter=starter, template=template.format(a=termA, b=termB))
    else:
        question = structure_template.format(template=template.format(a=termA, b=termB))
    
    # Select answer format and operation symbol
    answer_format = random.choice(answer_formats)
    op_symbol = random.choice(op_symbols[mode])
    
    # Format answer
    answer = answer_format.format(a=termA, b=termB, op=op_symbol, result=result)
    
    # Clean up any formatting issues
    question = question.replace("  ", " ").strip()
    answer = answer.replace("  ", " ").strip()
    
    # Ensure proper punctuation
    if not any(question.endswith(p) for p in ['.', '?', '!']):
        if '?' in question or structure_type in ['direct_question', 'conversational_help']:
            question += '?'
        else:
            question += '.'
    
    # Remove any double punctuation
    if question.endswith('..'):
        question = question[:-1]
    if question.endswith('??'):
        question = question[:-1]
    
    return [question, answer]

# Example usage for batch generation
def generate_dataset(num_samples: int, modes: List[str] = None, sm_mode=0) -> List[Tuple[str, str]]:
    """Generate a dataset of Q/A pairs."""
    if modes is None:
        modes = ["ADD", "SUB", "MULT", "DIVIDE"]
    
    dataset = []
    for _ in range(num_samples):
        mode = random.choice(modes)
        q, a = generate_q_a_pair(mode, sm_mode)
        dataset.append((q, a))
    
    return dataset

# Generate sample dataset
if __name__ == "__main__":
    samples = generate_dataset(20, None, 4)
    for i, (q, a) in enumerate(samples):
        #print(f"Sample {i+1}:")
        print(f"Q: {q}")
        print(f"A: {a}")
        print()
    samples = generate_dataset(80, None, 3)
    for i, (q, a) in enumerate(samples):
        #print(f"Sample {i+1}:")
        print(f"Q: {q}")
        print(f"A: {a}")
        print()
    samples = generate_dataset(200, None, 2)
    for i, (q, a) in enumerate(samples):
        #print(f"Sample {i+1}:")
        print(f"Q: {q}")
        print(f"A: {a}")
        print()
    samples = generate_dataset(400, None, 1)
    for i, (q, a) in enumerate(samples):
        #print(f"Sample {i+1}:")
        print(f"Q: {q}")
        print(f"A: {a}")
        print()
    samples = generate_dataset(800, None, 0)
    for i, (q, a) in enumerate(samples):
        #print(f"Sample {i+1}:")
        print(f"Q: {q}")
        print(f"A: {a}")
        print()