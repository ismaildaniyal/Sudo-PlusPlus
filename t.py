import streamlit as st

st.title("Pseudo-code to C++ Code Generator")

# Input fields for pseudo-code
pseudo_code = st.text_area("Enter your pseudo-code here", height=200)

# Function to generate C++ code from pseudo-code
def generate_cpp_code(pseudo_code):
    cpp_code = ""

    if "START" in pseudo_code:
        cpp_code += "#include <iostream>\nusing namespace std;\nint main() {\n"
    
    if "DECLARE" in pseudo_code:
        if "int" in pseudo_code:
            cpp_code += "    int variable;\n"
        elif "float" in pseudo_code:
            cpp_code += "    float variable;\n"
        elif "string" in pseudo_code:
            cpp_code += "    string variable;\n"

    if "SET" in pseudo_code and "to" in pseudo_code:
        cpp_code += "    variable = 10;\n"
    
    if "PRINT" in pseudo_code:
        cpp_code += '    cout << "Output: " << variable << endl;\n'
    
    if "READ" in pseudo_code:
        cpp_code += "    cin >> variable;\n"
    
    if "IF" in pseudo_code:
        cpp_code += "    if (condition) {\n"
    
    if "ELSE" in pseudo_code:
        cpp_code += "    } else {\n"
    
    if "ENDIF" in pseudo_code:
        cpp_code += "    }\n"
    
    if "LOOP" in pseudo_code:
        cpp_code += "    for (int i = 0; i < n; i++) {\n"
    
    if "ENDLOOP" in pseudo_code:
        cpp_code += "    }\n"
    
    if "END" in pseudo_code:
        cpp_code += "    return 0;\n}"

    return cpp_code

if st.button("Generate C++ Code"):
    if not pseudo_code:
        st.error("Please enter some pseudo-code.")
    else:
        generated_code = generate_cpp_code(pseudo_code)
        st.code(generated_code, language="cpp")
