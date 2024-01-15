# pythia-1b_Finetuning

Il dataset "**helicopters_data_complete.json**" è stato creato con ChatGPT 4, in totale sono 8388 esempi strutturati nel seguente modo:
The dataset  "helicopters_data_complete.json" was created by ChatGPT 4.
It consists in 8000-9000 Question/Answer about Helicopters.
An example is this one:
```json
{
    "instruction": "How much does a luxury helicopter cost?",
    "response": "A luxury helicopter, fitted with high-end features and advanced systems, can cost anywhere between $5 million and $20 million.",
    "context": "Enquiry about the cost of high-end luxury helicopters.",
    "category": "Helicopters Price"
}
```

Below you will find some responses returned by the finetuned model compared to the responses from chatGPT 3.5.
```
Question:What is the history of helicopters, and who is considered the father of modern helicopters?
Answer finetuned: The history of helicopters dates back to the early 20th century. Igor Sikorsky is often credited with inventing the first successful helicopter in 1939.
ChatGPT 3.5 response: The concept of helicopters dates back to the 15th century, but Igor Sikorsky is often credited as the father of modern helicopters, having designed and built the first practical one in the 1930s
Question:What are the main applications of helicopters in modern society?
Answer finetuned: Helicopters are widely used in modern society for purposes such as search and rescue, military operations, law enforcement, and air ambulance services.
ChatGPT 3.5 response:  Helicopters are widely used for purposes such as emergency medical services, search and rescue, transportation of goods, tourism, and military operations
Question:How does the propulsion system of a helicopter work, and what are its main components?
Answer finetuned: The propulsion system of a helicopter is made up of the engine, rotor system, and the fuel tank. They all work together to generate lift.
ChatGPT 3.5 response: Helicopters use rotor blades and an engine to generate lift. The main components include the main rotor, tail rotor, engine, and transmission system.
Question:What are the common challenges associated with using helicopters in civil and military operations?
Answer finetuned: Common challenges with using helicopters in civil and military operations include high operational costs, limited operating times, and the need for specific training.
ChatGPT 3.5 response: Challenges include high operating costs, limited range, noise pollution, and the need for skilled pilots. Military challenges include vulnerability to enemy fire.
Question:How do helicopters differ from fixed-wing aircraft in terms of design and operation?
Answer finetuned: Helicopters are more versatile as they can hover, takeoff and land vertically, and fly to airports.
ChatGPT 3.5 response: Helicopters can hover and take off vertically, thanks to their rotor system, while fixed-wing aircraft require a runway for takeoff and landing.
```
