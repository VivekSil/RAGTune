import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const Container = styled.div`
  display: flex;
  height: 100vh;
  background-color: #69f0ba;  /* Light shade of blue-green */
  font-family: Arial, sans-serif;
`;

const LeftPanel = styled.div`
  width: 25%;
  padding: 20px;
  background-color: #022133;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const CenterPanel = styled.div`
  width: 50%;
  padding: 20px;
  background-color: #022133;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
`;

const RightPanel = styled.div`
  width: 25%;
  padding: 20px;
  background-color: #022133;
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const SectionTitle = styled.h3`
  margin: 0;
  color: #333333;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #d9f9e1; /* Light background color for titles */
  padding: 10px;
  border-radius: 4px;
`;

const MainTitle = styled.h1`
  margin: 0;
  color: #333333;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #d9f9e1; /* Light background color for titles */
  padding: 10px;
  border-radius: 4px;
`;

const InputContainer = styled.div`
  padding: 10px;
  border: 1px solid #cccccc;
  border-radius: 4px;
  background-color: #fdfde3; /* Very light shade of yellow */
`;

const Input = styled.input`
  padding: 10px;
  border: none;
  border-radius: 4px;
  width: 100%;
  box-sizing: border-box;
  margin-bottom: 10px;
`;

const Dropdown = styled.select`
  padding: 10px;
  border: 1px solid #cccccc;
  border-radius: 4px;
  background-color: #fdfde3; /* Very light shade of yellow */
  margin-bottom: 10px;
`;

const Button = styled.button`
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  background-color: #007bff;
  color: white;
  cursor: pointer;
  width: 100%;
  box-sizing: border-box;

  &:hover {
    background-color: #0056b3;
  }
`;

const ChatBox = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 10px;
  overflow-y: auto;
  background-color: #fdfde3; /* Very light shade of yellow */
`;

const Message = styled.div`
  border-radius: 4px;
  padding: 10px;
  margin-bottom: 10px;
  align-self: ${(props) => (props.isUser ? 'flex-end' : 'flex-start')};
  background-color: ${(props) => (props.isUser ? '#d1e7dd' : '#f8d7da')};
`;

const ChatInputContainer = styled.div`
  display: flex;
  margin-top: 10px;
`;

const ChatInput = styled.input`
  flex: 1;
  padding: 10px;
  border: 1px solid #cccccc;
  border-radius: 4px;
  margin-right: 10px;
  background-color: #fdfde3; /* Very light shade of yellow */
`;

const ChatButton = styled.button`
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  background-color: #007bff;
  color: white;
  cursor: pointer;

  &:hover {
    background-color: #0056b3;
  }
`;

const ToggleButton = styled.button`
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  background-color: #007bff;
  color: white;
  cursor: pointer;

  &:hover {
    background-color: #0056b3;
  }
`;

const LoadSectionTitle = styled.h3`
  margin-top: 40px;
  color: #333333;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #d9f9e1; /* Light background color for titles */
  padding: 10px;
  border-radius: 4px;
`;

const App = () => {
  const [messages, setMessages] = useState([
    // { text: 'Hi there!', isUser: false },
    // { text: 'Hello! How can I help you?', isUser: true },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [loadMode, setLoadMode] = useState(false);
  const [showFileUpload, setShowFileUpload] = useState(true);
  const [showTextInput, setShowTextInput] = useState(true);
  const [showUrlInput, setShowUrlInput] = useState(true);
  const [selectedOption1, setSelectedOption1] = useState('');
  const [selectedOption2, setSelectedOption2] = useState('');

  const handleSendMessage = async () => {
    if (inputValue.trim() !== '') {
      setMessages([...messages, { text: inputValue, isUser: true }]);
      // const query = inputValue;
      const response = await axios.get('http://localhost:8000/api/generate', { params: { query: inputValue } });
      setMessages([...messages, { text: inputValue, isUser: true }, { text: response.data.response, isUser: false }]);
      setInputValue('');
    }
  };

  const handleLoadMessage = async () => {
    if (inputValue.trim() !== '') {
      setMessages([...messages, { text: inputValue, isUser: true }]);
      // const query = inputValue;
      const response = await axios.get('http://localhost:8000/api/rapt_generate', { params: { query: inputValue } });
      setMessages([...messages, { text: inputValue, isUser: true }, { text: response.data.response, isUser: false }]);
      setInputValue('');
    }
  };

  const toggleLoadMode = () => {
    setLoadMode(!loadMode);
  };

  const handleFileUpload = async (e) => {
    const formData = new FormData();
    formData.append('file', e.target.files[0]);
    await axios.post('http://localhost:8000/api/file_upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  };

  const handleFileUploadSubmit = async (e) => {
    e.preventDefault();
    const fileInput = e.target.elements.fileInput;
    if (fileInput.files.length > 0) {
      handleFileUpload({ target: { files: fileInput.files } });
    }
  };

  const handleTextInputSubmit = async (e) => {
    e.preventDefault();
    const text = e.target.elements.textInput.value;
    console.log(text)
    await axios.get('http://localhost:8000/api/rapt_tune', { params: { text } });
  };

  const handleModelInputSubmit = async (e) => {
    e.preventDefault();
    const model = e.target.elements.textInput.value;
    console.log(model )
    await axios.get('http://localhost:8000/api/embedding_finetuning', { params: { model } });
  };

  const handleUrlInputSubmit = async (e) => {
    e.preventDefault();
    const url = e.target.elements.urlInput.value;
    await axios.get('http://localhost:8010/api/get_url', { params: { url } });
  };

  // const handleDropdown1Change = async (e) => {
  //   setSelectedOption1(e.target.value);
  //   await axios.get('http://localhost:8010/api/rapt_tune', { params: { option: e.target.value } });
  // };

  // const handleDropdown2Change = async (e) => {
  //   setSelectedOption2(e.target.value);
  //   await axios.get('http://localhost:8010/api/ddp2', { params: { option: e.target.value } });
  // };

  const toggleFileUpload = () => {
    setShowFileUpload(!showFileUpload);
  };

  const toggleTextInput = () => {
    setShowTextInput(!showTextInput);
  };

  const toggleUrlInput = () => {
    setShowUrlInput(!showUrlInput);
  };

  return (
    <Container>
      <LeftPanel>
        <MainTitle>RAGTune</MainTitle>

        <SectionTitle onClick={toggleFileUpload}>
          File Upload {showFileUpload ? '▼' : '▲'}
        </SectionTitle>
        {showFileUpload && (
          <InputContainer>
            <form onSubmit={handleFileUploadSubmit}>
              <Input type="file" name="fileInput" />
              <Button type="submit">Upload</Button>
            </form>
          </InputContainer>
        )}

        {/* <SectionTitle onClick={toggleTextInput}>
          Text Input {showTextInput ? '▼' : '▲'}
        </SectionTitle>
        {showTextInput && (
          <InputContainer>
            <form onSubmit={handleTextInputSubmit}>
              <Input type="text" name="textInput" placeholder="Enter text..." />
              <Button type="submit">Submit</Button>
            </form>
          </InputContainer>
        )} */}

        <SectionTitle onClick={toggleUrlInput}>
          Connect using URL (Coming soon) {showUrlInput ? '▼' : '▲'}
        </SectionTitle>
        {/* {showUrlInput && (
          <InputContainer>
            <form onSubmit={handleUrlInputSubmit}>
              <Input type="url" name="urlInput" placeholder="Enter URL..." />
              <Button type="submit">Submit</Button>
            </form>
          </InputContainer>
        )} */}
      </LeftPanel>

      <CenterPanel>
        <SectionTitle>Chat Section</SectionTitle>
        <ChatBox>
          {messages.map((message, index) => (
            <Message key={index} isUser={message.isUser}>
              {message.text}
            </Message>
          ))}
        </ChatBox>
        <ChatInputContainer>
          <ChatInput
            type="text"
            name="chatInput"
            placeholder="Type a message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                loadMode ? handleLoadMessage() : handleSendMessage();
              }
            }}
          />
          <ChatButton onClick={loadMode ? handleLoadMessage : handleSendMessage}>
            Send
          </ChatButton>
          
         </ChatInputContainer>
    </CenterPanel>

      <RightPanel>
      <MainTitle>Tuning methods</MainTitle>
      <SectionTitle onClick={toggleTextInput}>
          1. Prompt Tuning {showTextInput ? '▼' : '▲'}
        </SectionTitle>
        {showTextInput && (
          <InputContainer>
            <form onSubmit={handleTextInputSubmit}>
              <Input type="text" name="textInput" placeholder="Enter instructions..." />
              <Button type="submit">Submit</Button>
            </form>
          </InputContainer>
        )}
        <SectionTitle onClick={toggleTextInput}>
          2. Embedding  Fine-tuning{showTextInput ? '▼' : '▲'}
        </SectionTitle>
        {showTextInput && (
          <InputContainer>
            <form onSubmit={handleModelInputSubmit}>
              <Input type="text" name="textInput" placeholder="Enter model name" />
              <Button type="submit">Submit</Button>
            </form>
          </InputContainer>
        )}
        <SectionTitle onClick={toggleTextInput}>
          3. LLM Fine-tuning (Coming soon) {showTextInput ? '▼' : '▲'}
        </SectionTitle>

        {/* <SectionTitle>Dropdown 1</SectionTitle>
        <Dropdown value={selectedOption1} onChange={handleDropdown1Change}>
          <option value="">Select an option</option>
          <option value="Option 1">Option 1</option>
          <option value="Option 2">Option 2</option>
          <option value="Option 3">Option 3</option>
        </Dropdown> */}
        {/* <InputContainer>
            <form onSubmit={handleTextInputSubmit}>
              <Input type="text" name="textInput" placeholder="Enter text..." />
              <Button type="submit">Submit</Button>
            </form>
          </InputContainer> */}
        {/* <Button onClick={() => handleDropdown1Change({ target: { value: selectedOption1 } })}>Submit</Button> */}

        {/* <SectionTitle>Dropdown 2</SectionTitle>
        <Dropdown value={selectedOption2} onChange={handleDropdown2Change}>
          <option value="">Select an option</option>
          <option value="Option A">Option A</option>
          <option value="Option B">Option B</option>
          <option value="Option C">Option C</option>
        </Dropdown> */}
        
        {/* <Button onClick={() => handleDropdown2Change({ target: { value: selectedOption2 } })}>Submit</Button> */}

        <LoadSectionTitle>Optimized program
        <ToggleButton onClick={toggleLoadMode} style={{ color: '#ffffff' }}>
          {loadMode ? 'Unload' : 'Load'}
        </ToggleButton>
        </LoadSectionTitle>
        <SectionTitle>Select model (Coming soon)  </SectionTitle>
      </RightPanel>
    </Container>
  );
};

export default App;
