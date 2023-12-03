import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

import '../utils/text.form.global.dart';
import 'buttons/button.dart';

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({Key? key}) : super(key: key);

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordViewState();
}

class _ForgotPasswordViewState extends State<ForgotPasswordScreen> {
  final TextEditingController emailController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SingleChildScrollView(
        child: SafeArea(
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.all(15.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 50),

                const Text(
                  'Forgot your password?',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),

                const SizedBox(height: 15),

                //// Email Input
                TextFormGlobal(
                  controller: emailController,
                  text: 'input email to verify',
                  obscure: false,
                  textInputType: TextInputType.emailAddress,
                ),

                const SizedBox(
                  height: 10,
                ),
                button(context, "RESET PASSWORD", () async {
                  await FirebaseAuth.instance
                      .sendPasswordResetEmail(
                          email: emailController.text.toString().trim())
                      .then((value) => Navigator.of(context).pop());
                }),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
